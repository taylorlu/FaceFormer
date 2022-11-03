import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

def trainer(args, train_loader, model, optimizer, epoch=100):
    flame_mask = pickle.load(open(args.flame_mask, 'rb'), encoding='latin1')
    # lip_mask = torch.ones([5023, 3]).to(device="cuda")
    # lip_mask[flame_mask['lips'], :] = 10.
    # lip_mask = torch.reshape(lip_mask, (1, -1))
    right_eye_region = flame_mask['right_eye_region']
    left_eye_region = flame_mask['left_eye_region']
    face = flame_mask['face']

    eye_region = set(right_eye_region).union(set(left_eye_region))
    face_except_eye_region_and_ball = list(set(face).difference(eye_region))
    keep_mask = torch.ones([args.vertice_dim, 3]).to(torch.device(args.device))
    keep_mask[face_except_eye_region_and_ball, :] = 10.
    keep_mask = torch.reshape(keep_mask, (1, -1))

    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, template,  vertice, one_hot, keep_mask, teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if iteration % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))

        if (e > 0 and e % 50 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path,'{}_model.pth'.format(e)))

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="owndata", help='vocaset or BIWI')
    parser.add_argument("--exp_jaw_dim", type=int, default=53, help='number of exp jaw coeff, 50 + 3')
    parser.add_argument("--vertice_dim", type=int, default=5023, help='number of vertices - 5023 for vocaset')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--max_seq_len", type=int, default=300, help='max_seq_len')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--flame_mask", type=str, default="models/FLAME_masks.pkl", help='vertices mask of flame model')
    parser.add_argument("--facebook", type=bool, default=False, help='facebook wav2vec or tencent wav2vec, default false/tencent')
    args = parser.parse_args()

    #load data
    dataset, subjects_list = get_dataloaders(args)

    os.makedirs(args.save_path, exist_ok=True)
    pickle.dump(subjects_list, open(os.path.join(args.save_path, 'subject_list.pkl'), 'wb'))

    #build model
    model = Faceformer(args, subjects_list)
    print("model parameters: ", count_parameters(model))

    pths = os.listdir(args.save_path)
    pths = [pth for pth in pths if pth.endswith('.pth')]
    if(len(pths)!=0):
        pths = [int(x.split('_')[0]) for x in pths]
        pths.sort()
        checkpoint = os.path.join(args.save_path, f'{pths[-1]}_model.pth')
        model.load_state_dict(torch.load(checkpoint))
        print(f'restore from {checkpoint}')

    # to cuda
    model = model.to(torch.device(args.device))

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], model, optimizer, epoch=args.max_epoch)

if __name__=="__main__":
    main()
