import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'egl' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh

@torch.no_grad()
def test_model(args):
    speaker_len = 23
    #build model
    model = Faceformer(args, speaker_len)
    model.load_state_dict(torch.load(args.model_name))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    one_hot_labels = np.eye(speaker_len)
    one_hot = one_hot_labels[args.choice]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = list(templates.values())[0]
             
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    if(args.facebook):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    else:
        processor = Wav2Vec2FeatureExtractor.from_pretrained('TencentGameMate/chinese-wav2vec2-base')
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    vertice_out, exp_jaw_out = model.predict(audio_feature, template, one_hot, args.fps)
    vertice_out = vertice_out.squeeze() # (seq_len, V*3)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, 'exp_jaw.npy'), exp_jaw_out.cpu().numpy())
    render_sequence(args, vertice_out)

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    # light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())

    # light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence(args, vertice_out):
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    template_file = os.path.join("models/FLAME_sample.ply")

    print("rendering: ", test_name)
                 
    template = Mesh(filename=template_file)
    verts = np.reshape(vertice_out.cpu().numpy(), (-1,args.vertice_dim,3))

    output_path = args.output_path

    num_frames = verts.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(verts[0], axis=0)

    for i_frame in range(num_frames):
        render_mesh = Mesh(verts[i_frame], template.f)
        pred_img = render_mesh_helper(args,render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()

    video_fname = os.path.join(output_path, test_name+'.mp4')
    cmd = ('ffmpeg' + ' -r 30 -i {0} -i {1} -pix_fmt yuv420p -qscale 0 {2}'.format(
       tmp_video_file.name, wav_path, video_fname)).split()
    call(cmd)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="save/1000_model.pth")
    parser.add_argument("--dataset", type=str, default="owndata", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--max_seq_len", type=int, default=600, help='max_seq_len')
    parser.add_argument("--vertice_dim", type=int, default=5023, help='number of vertices - 5023 for vocaset')
    parser.add_argument("--exp_jaw_dim", type=int, default=53, help='number of exp jaw coeff, 50 + 3')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--choice", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="wav_clips/1.wav", help='path of the input audio signal')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--facebook", type=bool, default=False, help='facebook wav2vec or tencent wav2vec, default false/tencent')
    args = parser.parse_args()   

    test_model(args)

if __name__=="__main__":
    main()
