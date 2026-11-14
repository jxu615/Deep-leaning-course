# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:54:25 2022

@author: 13572
"""
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
from .dataset_real import Dataset
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
            psf = psf.repeat(groups, 1, 1, 1)
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=groups)
        return output_tensor

class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio

        psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        self.psf = nn.Parameter(psf)

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
        self.S1_lr = args.S1_lr
        self.niter1=args.niter1
        self.ker_size = self.args.scale_factor  #8
        self.ratio    = self.args.scale_factor
        self.hs_bands = self.tensor_lr_hsi.size(1)
        self.ms_bands = self.tensor_hr_msi.size(1)
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.S1_lr)


    def train(self,  verb=True):


        lr_hsi, hr_msi = self.tensor_lr_hsi.to(device), self.tensor_hr_msi.to(device)
        for epoch in range(1, self.niter1 + 1):
            lr_msi_fhsi_est, lr_msi_fmsi_est = self.model(lr_hsi, hr_msi)

            loss = torch.sum(torch.abs(lr_msi_fhsi_est - lr_msi_fmsi_est))
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


                        print('************')

        PATH = os.path.join(self.args.expr_dir,self.model.__class__.__name__ + '.pth')
        os.makedirs(self.args.expr_dir, exist_ok=True)
        torch.save(self.model.state_dict(), PATH)
        return lr_msi_fhsi_est, lr_msi_fmsi_est


    def get_save_result(self, is_save=True):


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

            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)
            psf_div = torch.div(1.0, psf_div)
            w.mul_(psf_div)

        if hasattr(model, 'srf'):

            w = model.srf.data
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)
            w.mul_(srf_div)


