#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
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

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        pc1, pc2, color1, color2, flow, mask1 = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)
        loss_1 = torch.mean(mask1 * torch.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)

        pc1, pc2 = pc1.permute(0,2,1), pc2.permute(0,2,1)
        pc1_ = pc1 + flow_pred

        total_loss += loss_1.item() * batch_size
        

    return total_loss * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    num_examples = 0
    total_loss = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        pc1, pc2, color1, color2, flow, mask1 = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda().transpose(2,1).contiguous()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)
        loss_1 = torch.mean(mask1 * torch.sum((flow_pred - flow) ** 2, 1) / 2.0)

        pc1, pc2, flow_pred = pc1.permute(0,2,1), pc2.permute(0,2,1), flow_pred.permute(0,2,1)
        pc1_ = pc1 + flow_pred

        loss_1.backward()

        opt.step()
        total_loss += loss_1.item() * batch_size

        # if (i+1) % 100 == 0:
        #     print("batch: %d, mean loss: %f" % (i, total_loss / 100 / batch_size))
        #     total_loss = 0
    return total_loss * 1.0 / num_examples


def test(args, net, test_loader, boardio, textio):

    test_loss = test_one_epoch(args, net, test_loader)

    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f'%test_loss)


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    best_test_loss = np.inf
    for epoch in range(args.epochs):
        scheduler.step()
        textio.cprint('==epoch: %d=='%epoch)
        train_loss = train_one_epoch(args, net, train_loader, opt)
        textio.cprint('mean train EPE loss: %f'%train_loss)

        test_loss = test_one_epoch(args, net, test_loader)
        textio.cprint('mean test EPE loss: %f'%test_loss)

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f'%test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        boardio.add_scalar('Train Loss', train_loss, epoch+1)
        boardio.add_scalar('Test Loss', test_loss, epoch+1)
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)
        
        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_flownet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='flownet', metavar='N',
                        choices=['flownet'],
                        help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Point Number [default: 2048]')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--dataset', type=str, default='SceneflowDataset',
                        choices=['SceneflowDataset'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data_processed_maxcut_35_20k_2k_8192', metavar='N',
                        help='dataset to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--pretrained', type=str, default='', metavar='N',
                        help='Pretrained model path')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'SceneflowDataset':
        train_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    if args.model == 'flownet':
        net = FlowNet3D().cuda()
        net.apply(weights_init)
        if args.pretrained:
            net.load_state_dict(torch.load(args.pretrained), strict=False)
            print("Pretrained Model Loaded Successfully!")
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    # boardio.close()


if __name__ == '__main__':
    main()