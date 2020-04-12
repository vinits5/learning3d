#!/usr/bin/env python
# -*- coding: utf-8 -*-


import open3d as o3d
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from learning3d.models import FlowNet3D
from learning3d.data_utils import SceneflowDataset
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source + np.array([0,0.5,0.5]))
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    template_.paint_uniform_color([1, 0, 0])
    source_.paint_uniform_color([0, 1, 0])
    transformed_source_.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([template_, source_, transformed_source_])

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    for i, data in enumerate(tqdm(test_loader)):
        data = [d.to(args.device) for d in data]
        pc1, pc2, color1, color2, flow, mask1 = data
        pc1 = pc1.transpose(2,1).contiguous()
        pc2 = pc2.transpose(2,1).contiguous()
        color1 = color1.transpose(2,1).contiguous()
        color2 = color2.transpose(2,1).contiguous()
        flow = flow
        mask1 = mask1.float()

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)
        loss_1 = torch.mean(mask1 * torch.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)

        pc1, pc2 = pc1.permute(0,2,1), pc2.permute(0,2,1)
        pc1_ = pc1 - flow_pred
        print("Loss: ", loss_1)
        display_open3d(pc1.detach().cpu().numpy()[0], pc2.detach().cpu().numpy()[0], pc1_.detach().cpu().numpy()[0])
        total_loss += loss_1.item() * batch_size        

    return total_loss * 1.0 / num_examples


def test(args, net, test_loader):
    test_loss = test_one_epoch(args, net, test_loader)

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--model', type=str, default='flownet', metavar='N',
                        choices=['flownet'], help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Point Number [default: 2048]')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--dataset', type=str, default='SceneflowDataset',
                        choices=['SceneflowDataset'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data_processed_maxcut_35_20k_2k_8192', metavar='N',
                        help='dataset to use')
    parser.add_argument('--pretrained', type=str, default='learning3d/pretrained/exp_flownet/models/model.best.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda')

    if args.dataset == 'SceneflowDataset':
        test_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    net = FlowNet3D()
    assert os.path.exists(args.pretrained), "Pretrained Model Doesn't Exists!"
    net.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    net = net.to(args.device)
        
    test(args, net, test_loader)
    print('FINISH')


if __name__ == '__main__':
    main()