import os
import torch
from flame.FLAME import FLAME
import numpy as np
import shutil, pickle

flame_model_path = os.path.join('models', 'generic_model-2020.pkl')
flame_lmk_embedding_path = os.path.join('models', 'flame_static_embedding.pkl.pkl')
flame = FLAME(flame_model_path, flame_lmk_embedding_path).to(torch.device("cuda"))

prediction = torch.tensor(np.zeros([1, 53])).to(device="cuda")

pose_params = torch.zeros([prediction.shape[0], 15]).to(device="cuda")
pose_params[:, 6:9] = prediction[:, 50:]
expression_params = torch.zeros([prediction.shape[0], 100]).to(device="cuda")
expression_params[:, :50] = prediction[:, :50]
shape_params = torch.zeros([1, 300]).to(device="cuda")

verts, landmarks3d = flame(1, \
                            shape_params=shape_params, \
                            expression_params=expression_params, \
                            pose_params=pose_params)
verts = np.array(verts.cpu().numpy().reshape(verts.shape[0], -1))

pkl = {}

os.makedirs('owndata/vertices_npy',  exist_ok=True)
os.makedirs('owndata/wav',  exist_ok=True)

root_path = '/mnt/audio_exp_dataset'
speakers = os.listdir(root_path)
for speaker in speakers:
    sentences = os.listdir(os.path.join(root_path, speaker))
    for sentence in sentences:
        wav_path = os.path.join(root_path, speaker, sentence, 'audio.wav')
        exp_jaw = os.path.join(root_path, speaker, sentence, 'exp_jaw.npy')

        prediction = torch.tensor(np.load(exp_jaw)).to(device="cuda")

        pose_params = torch.zeros([prediction.shape[0], 15]).to(device="cuda")
        pose_params[:, 6:9] = prediction[:, 50:]
        expression_params = torch.zeros([prediction.shape[0], 100]).to(device="cuda")
        expression_params[:, :50] = prediction[:, :50]
        shape_params = torch.zeros([1, 300]).to(device="cuda")

        verts, landmarks3d = flame(pose_params.shape[0], \
                                    shape_params=shape_params, \
                                    expression_params=expression_params, \
                                    pose_params=pose_params)

        verts = np.array(verts.cpu().numpy().reshape(verts.shape[0], -1))
        print(verts.shape)
        np.save('owndata/vertices_npy/{}_{}.npy'.format(speaker, sentence), verts)
        shutil.copy(wav_path, 'owndata/wav/{}_{}.wav'.format(speaker, sentence))
    
    pkl[speaker] = verts[0, ...]
    print(verts.shape)

pickle.dump(pkl, open('owndata/templates.pkl', 'wb'))
