# -*- coding: utf-8 -*-


import argparse
import torch
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
###通用参数
parser.add_argument('--scale_factor',type=int,default=10, help='缩放尺度')
parser.add_argument('--sp_root_path',type=str, default='data/spectral_response/',help='光谱相应地址')
parser.add_argument('--default_datapath',type=str, default="data/",help='高光谱读取地址')
parser.add_argument('--data_name',type=str, default="CAVE",help='Houston18=8 DC=10 TG=12 Chikusei=16')
parser.add_argument("--device", type=str, default='0', help='')
parser.add_argument("--gpu_ids", type=str, default='0', help='')
parser.add_argument('--checkpoints_dir',type=str, default='checkpoints',help='高光谱读取地址')
parser.add_argument('--lambda_A', type=float, default=10, help='')
parser.add_argument('--lambda_B', type=float, default=10, help='')
parser.add_argument('--lambda_D', type=float, default=0.01, help='weight for sum2one and sparse   mu')
parser.add_argument('--lambda_F', type=float, default=1000, help='weight for lrmsi gamma')
parser.add_argument('--lambda_G', type=float, default=1, help='weight for ')
####
parser.add_argument('--simulate',type=int, default=1,help='是否使用模拟数据 No=0 Yes=1')
parser.add_argument("--fusion_weight", type=float, default=0.5,help='default=0.5')
parser.add_argument("--num_s", type=int, default=160,help='')
parser.add_argument('--block', type=int, default=2)
parser.add_argument("--head", type=int, default=2)
parser.add_argument("--window", type=int, default=4)
parser.add_argument('--freq', type=int, default=20, help='20')
#parser.add_argument("--select", type=str, default=False,help='是否在第二阶段采用 退化引导融合策略')

#训练参数
#训练srf学习率
parser.add_argument("--S1_lr", type=float, default=0.001,help='0.001')
parser.add_argument('--niter1', type=int, default=5000, help='# 3000of iter at starting learning rate3000')
#MoiT学习率
parser.add_argument("--lr", type=float, default=3e-3,help='学习率')
parser.add_argument('--niter', type=int, default=100, help='#2000 of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# 2000of iter to linearly decay learning rate to zero')
args=parser.parse_args()

#device = torch.device(  'cuda:{}'.format(args.gpu_ids[0])  ) if args.gpu_ids else torch.device('cpu') 
device = torch.device( 'cuda:{}'.format(args.gpu_ids) ) if  torch.cuda.is_available() else torch.device('cpu')
args.device=device
args.sigma = args.scale_factor / 2.35482

args.expr_dir=os.path.join('checkpoints', args.data_name+'SF'+str(args.scale_factor)+'lambda_G'+str(args.lambda_G)+'new')
#opt.sigma = scale_factor / 2.35482 checkpoints_dir