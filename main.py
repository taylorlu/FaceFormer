import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

def trainer(args, train_loader, model, optimizer, criterion, epoch=100):
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, exp_jaw, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, exp_jaw, one_hot  = audio.to(device="cuda"), exp_jaw.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, exp_jaw, one_hot, criterion, teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if iteration % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))

        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="owndata", help='vocaset or BIWI')
    parser.add_argument("--exp_jaw_dim", type=int, default=53, help='number of exp jaw coeff, 50 + 3')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--test_wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    args = parser.parse_args()
    
    #load data
    dataset, speaker_len = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    #build model
    model = Faceformer(args, speaker_len)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], model, optimizer, criterion, epoch=args.max_epoch)

if __name__=="__main__":
    main()
