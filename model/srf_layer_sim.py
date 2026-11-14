# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:54:25 2022

@author: 13572
"""
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as fun
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
#from .read_data import readdata
#from .read_data_data import readdata
from .dataset import Dataset
#import evaluation

import random
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlurDown(object):
    def __init__(self):
        #self.shift_h = shift_h
        #self.shift_w = shift_w
        #self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, groups, ratio):
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1) #8X1X8X8

        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=groups) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):  # PSF为#1 1 ratio ratio 大小的tensor
        _, C, _, _ = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)  # 8X1X8X8
            # input_tensor: 1X8X400X400
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),
                                   groups=C)  # ratio为步长 None代表bias为0，padding默认为无
        return output_tensor

class SRF_up():
    def __call__(self, input_tensor, srf):
        srf = srf.permute(1,0,2,3)
        output_tensor = fun.conv2d(input_tensor, srf, None)
        return output_tensor

class SRF_down():
    def __call__(self, input_tensor, srf):
        output_tensor = fun.conv2d(input_tensor, srf, None)
        return output_tensor


class Discriminator(nn.Module):
    def __init__(self ,ratio):
        super(Discriminator, self).__init__()
        self.avp = torch.nn.AdaptiveAvgPool2d(1)
        self.srf_up = SRF_up()
        self.psf_down = PSF_down()
        self.ratio =ratio
        self.temperature = 0.1


    def forward(self,srf,psf,x,y):#x多光谱 y高光谱
        x = self.srf_up(x,srf)
        x1 = self.psf_down(x,psf,self.ratio)
        x_spa = x1
        y_spa = y
        x_spa = rearrange(x_spa, 'b c h w->  c (b h w)')
        y_spa = rearrange(y_spa, 'b c h w->  c (b h w)')
        similarity_matrix1 = torch.mm(F.normalize(x_spa, dim=1),
                                      F.normalize(y_spa, dim=1).T)
        # save_dir=r'E:\xj\FFT代码\checkpoints'
        # filename = 'sim.mat'
        # os.makedirs(save_dir, exist_ok=True)
        # # 保存为.mat文件
        # save_path = os.path.join(save_dir, filename)
        # sio.savemat(save_path, {'sim_matrix': similarity_matrix1.detach().cpu().numpy()})
        #print(similarity_matrix1)
        #正样本选择
        positives1 = torch.diag(similarity_matrix1)
        #负样本选择
        # min_pos_sim = positives1.min()
        # neg_mask = (similarity_matrix1 < min_pos_sim).float()
        # neg_mask.fill_diagonal_(0)  # 排除正样本自身

        neg_mask = torch.ones_like(similarity_matrix1, dtype=torch.bool)
        neg_mask.fill_diagonal_(0)  # 排除对角线（正样本）

        k = int(len(positives1)*0.1)
        topk_values, topk_indices = torch.topk(positives1, k=k)

        all_neg_sim = similarity_matrix1[neg_mask]  # 展平后的负样本相似度
        bottomk_values, bottomk_indices = torch.topk(-all_neg_sim, k=k)



        #exp_sim1 = torch.exp(similarity_matrix1 / self.temperature)
        numerator1 = torch.exp(topk_values / self.temperature)
        #exp_neg = (exp_sim1 * neg_mask).sum(dim=-1)
        exp_neg = torch.exp(bottomk_values / self.temperature)
        #denominator1 = exp_sim1.sum(dim=-1) - numerator1
        loss_spa = -torch.log(numerator1 / (numerator1+exp_neg)).mean()
        return loss_spa
class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size #8
        self.ratio = ratio #8


        psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        self.psf = nn.Parameter(psf)

        #srf = torch.rand([self.ms_bands, self.hs_bands, 1, 1]) #0-1均匀分布
        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands)
        self.srf = nn.Parameter(srf)
        self.blur_down = BlurDown()

    def forward(self, lr_hsi, hr_msi):


        srf_div = torch.sum(self.srf, dim=1, keepdim=True)


        srf_div = torch.div(1.0, srf_div)


        srf_div = torch.transpose(srf_div, 0, 1)


        lr_msi_fhsi = fun.conv2d(lr_hsi, self.srf, None)
        lr_msi_fhsi = torch.mul(lr_msi_fhsi, srf_div)
        lr_msi_fhsi = torch.clamp(lr_msi_fhsi, 0.0, 1.0)
        lr_msi_fmsi = self.blur_down(hr_msi, self.psf,  self.ms_bands, self.ratio)
        lr_msi_fmsi = torch.clamp(lr_msi_fmsi, 0.0, 1.0)
        return lr_msi_fhsi, lr_msi_fmsi


class Blind(Dataset):
    def __init__(self, args,isTrain=True):
        super().__init__(args)
        self.G = args.lambda_G

        self.S1_lr = args.S1_lr
        self.niter1 = args.niter1
        self.ker_size = self.args.scale_factor  #8
        self.ratio = self.args.scale_factor
        self.hs_bands = self.tensor_lr_hsi.size(1)
        self.ms_bands = self.tensor_hr_msi.size(1)
        self.Discriminator = Discriminator(self.ratio).to(device)
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.S1_lr)


    def train(self, verb=True):

        #hsi, msi = self.__hsi.cuda(), self.__msi.cuda()
        lr_hsi, hr_msi = self.tensor_lr_hsi.to(device), self.tensor_hr_msi.to(device)
        for epoch in range(1, self.niter1 + 1):
            lr_msi_fhsi_est, lr_msi_fmsi_est = self.model(lr_hsi, hr_msi)
            loss1 = self.Discriminator(self.model.srf, self.model.psf, hr_msi, lr_hsi)
            loss = torch.sum(torch.abs(lr_msi_fhsi_est - lr_msi_fmsi_est))+loss1*0
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()


            self.model.apply(self.check_weight)


            with torch.no_grad():
                if verb is True:
                    if (epoch ) % 100 == 0:
                        print("____________________________________________")
                        print('epoch: %s, lr: %s, loss: %s' % (epoch, self.S1_lr, loss))

                        lr_msi_fhsi_est_numpy=lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        lr_msi_fmsi_est_numpy=lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        self.lr_msi_fhsi_est_numpy=lr_msi_fhsi_est_numpy
                        self.lr_msi_fmsi_est_numpy=lr_msi_fmsi_est_numpy


        PATH = os.path.join(self.args.expr_dir,self.model.__class__.__name__ + '.pth')
        os.makedirs(self.args.expr_dir, exist_ok=True)
        torch.save(self.model.state_dict(), PATH)
        return lr_msi_fhsi_est, lr_msi_fmsi_est


    def get_save_result(self, is_save=True):

        # self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        psf = self.model.psf.data.cpu().detach().numpy()
        srf = self.model.srf.data.cpu().detach().numpy()
        psf = np.squeeze(psf)
        srf = np.squeeze(srf).T
        self.psf, self.srf = psf , srf
        if is_save is True:
            sio.savemat(os.path.join(self.args.expr_dir,'estimated_psf_srf.mat'), {'psf_est': psf, 'srf_est': srf})
        return

    @staticmethod
    def check_weight(model):

        if hasattr(model, 'psf'):
            #print(model,'psf')
            w = model.psf.data
            #w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)
            psf_div = torch.div(1.0, psf_div)
            w.mul_(psf_div)
            w.clamp_(0.0, 1.0)

        if hasattr(model, 'srf'):
            #print(model,'srf')
            w = model.srf.data # torch.Size([8, 46, 1, 1])
            srf_div = torch.sum(w, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)
            w.mul_(srf_div)
            w.clamp_(0.0, 1)

# if __name__ == "__main__":
#     from options.train_options import TrainOptions
#     opt = TrainOptions()
#     parser = opt.initialize(parser=None)
#     args = parser.parse_args()
