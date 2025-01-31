
"""
Modified code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master
"""
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from models.loss import criterion
from torch.utils.data import DataLoader

from models.pointnet import PointNetOpt
from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.tensorboard import SummaryWriter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--epoch', default=5001, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--data_path', type=str, default='../data/LV_left/train_data.pickle', help='data path to optimize')
    parser.add_argument('--tag', default='manual', help='tex or point')
    parser.add_argument('--sub_id', default= '', help='LBC subject id')

    return parser.parse_args()

def train_data(data_file):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
 
    # Convert vertices and faces to tensors and move to GPU
    vertices = torch.from_numpy(np.asarray(data['vertices']).astype(np.float32)[np.newaxis, :, :]).cuda()
    faces = torch.from_numpy(np.asarray(data['faces']).astype(np.int32)[np.newaxis, :, :]).cuda()
    target = torch.from_numpy(np.asarray(data['target']).astype(np.float32)[np.newaxis, :, :]).cuda()
    pred_mesh = Meshes(verts=list(vertices), faces=list(faces))

    return vertices, faces, target, pred_mesh
def create_directory(directory_path, log=False):
    try:
        # Create target Directory
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
        if log:
            raise IOError(f"The file '{directory_path}' exists.")
        else:
            pass

def main(args):
    '''LOG'''
    args = parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = args.batch_size
    # print(args.sub_id,"!!!!!")
    create_directory(f"./edinburgh/scan1/{args.tag}/log/{args.sub_id}", True)
    writer = SummaryWriter(log_dir=f"./edinburgh/scan1/{args.tag}/log/{args.sub_id}")
    
    '''MODEL LOADING'''
    model = PointNetOpt(num_classes=3, input_transform=False, feature_transform=False).to("cuda")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch//5, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    global_step = 0
    '''TRANING'''
    print('Start training...')
    model.train()
    # data preparation
    vertices, faces, target, pred_mesh = train_data(args.data_path)
    # Convert vertices and faces to tensors and move to GPU
    orig_vertices = vertices.clone()
    # Optimization loop
    for epoch in tqdm(range(args.epoch)):

        optimizer.zero_grad()

        vertices = pred_mesh.verts_list()[0].unsqueeze(0)

        # Forward pass
        pred = model(vertices)

        # Avoid inplace operations by creating new tensors
        verts = vertices.clone() + pred.transpose(2, 1).clone()
        l2_loss = loss_fn(orig_vertices, verts)
        loss, log = criterion(verts, target, faces)
        loss += l2_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        log['l2 loss'] =l2_loss
        if epoch % 100 == 0 : print(epoch, log)
        
        for key in log.keys():
            writer.add_scalar(key, log[key], global_step=epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch)
        if epoch % 500 == 0 or epoch == args.epoch-1:
            verts_np = verts.detach().cpu().numpy()
            create_directory(rf'./edinburgh/scan1/{args.tag}/out/{args.sub_id}')
            np.save(rf'./edinburgh/scan1/{args.tag}/out/{args.sub_id}/{epoch}_{args.sub_id}_{loss}.npy', verts_np)
    print('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)