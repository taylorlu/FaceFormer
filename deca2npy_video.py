import argparse, os, pickle
from demo import render_sequence
from flame.FLAME import FLAME
import torch
import numpy as np

cuda = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='DECA output sequence param_dict.pkl to normed npy and render to video with audio')
parser.add_argument("--input", type=str, help='sequence folder output by DECA')
parser.add_argument('--fps', type=float, help='frame rate of input video')
parser.add_argument("--wav_path", type=str, help='path of the input audio signal')
parser.add_argument("--output_path", type=str, help='folder of the rendered video sequences and npy')
parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset')
parser.add_argument("--vertice_dim", type=int, default=5023, help='number of vertices - 5023 for vocaset')
parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
args = parser.parse_args()

flame_model_path = os.path.join('models', 'generic_model-2020.pkl')
flame_lmk_embedding_path = os.path.join('models', 'flame_static_embedding.pkl')
flame = FLAME(flame_model_path, flame_lmk_embedding_path).to(cuda)

pkl = [pkl for pkl in os.listdir(args.input) if pkl.endswith('param_dict.pkl')][0]
pkl = os.path.join(args.input, pkl)
seq_params = pickle.load(open(pkl, 'rb'), encoding='latin1')

seq_len = len(seq_params['pose'])
to_len = round(seq_len*30/args.fps)
to_exp_jaw = np.zeros([to_len, 53])
for i in range(to_len):
    to_exp_jaw[i, :50] = seq_params['exp'][round(i*args.fps/30)][0, :]
    to_exp_jaw[i, 50:] = seq_params['pose'][round(i*args.fps/30)][0, 3:]

os.makedirs(args.output_path, exist_ok=True)
np.save(os.path.join(args.output_path, 'exp_jaw.npy'), to_exp_jaw)

shapes = torch.zeros([1, 300])
exps = torch.zeros([to_len, 100])
poses = torch.zeros([to_len, 15])
for i in range(to_len):
    exps[i, :50] = torch.tensor(to_exp_jaw[i, :50])
    poses[i, 6:9] = torch.tensor(to_exp_jaw[i, 50:])

basic_exp_embedding = torch.index_select(exps, dim=0, index=torch.argmin(torch.norm(exps, dim=1)))
basic_pose_embedding = torch.index_select(poses, dim=0, index=torch.argmin(torch.norm(poses[:, 6:9], dim=1)))

expression_params = exps - basic_exp_embedding
pose_params = poses - basic_pose_embedding

verts, landmarks3d = flame(pose_params.shape[0], \
                            shape_params=shapes.to(cuda), \
                            expression_params=expression_params.to(cuda), \
                            pose_params=pose_params.to(cuda))
print(verts.shape)
render_sequence(args, verts)
