# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 19:38:23 2021

@author: 13572
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
from .evaluation import MetricsCal
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy
from .dataset_real import Dataset
from . import network
import torch.nn.functional as F


# from.network_s2 import def_two_stream_split


class fus(Dataset):
    def __init__(self, args, srf):
        super().__init__(args)
        self.args = args
        self.hs_band = self.tensor_lr_hsi.shape[1]
        self.ms_band = self.tensor_hr_msi.shape[1]
        self.h = self.tensor_hr_msi.shape[2]
        self.l_h=self.tensor_lr_hsi.shape[2]
        self.n_r= self.h//self.args.window
        self.srf = srf
        self.lr_hsi = self.tensor_lr_hsi
        self.hr_msi = self.tensor_hr_msi

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - self.args.niter) / float(self.args.niter_decay + 1)
            return lr_l

        self.net_MSI2S = network.define_msi2s(h=self.h, n_r=self.n_r ,window_a=self.args.window,window_b=self.args.window,n_heads=self.args.head,n=self.args.block,
                                              input_ch=self.ms_band, output_ch=self.args.num_s,gpu_ids=self.args.device)
        # shared endmember (also represents decoder)
        self.net_s2img = network.define_s2img(input_ch=self.args.num_s, output_ch=self.hs_band, gpu_ids=self.args.device)

        #self.net_m2img = network.define_s2img(input_ch=num_s, output_ch=msi_channels, gpu_ids=self.gpu_ids)

        # encoder for hsi
        self.net_LR2s = network.define_lr2s(l_h=self.l_h , window_a=self.args.window,window_b=self.args.window,input_ch=self.hs_band,
                                            n_heads=self.args.head,n=self.args.block ,output_ch=self.args.num_s, gpu_ids=self.args.device)
        # define psf function
        self.net_PSF = network.define_psf(scale=self.args.scale_factor,gpu_ids=self.args.device)
        # reconstruct msi image from target HRHSI image
        self.net_G_HR2MSI = network.define_hr2msi(args=self.args,
                                                    sp_matrix=self.srf,
                                                    gpu_ids=self.args.device)
        self.net_Spacetransform = network.define_spatial_transform()

        self.criterionSumToOne = network.SumToOneLoss().to(self.args.device)
        self.criterionSparse = network.SparseKLloss().to(self.args.device)
        self.criterionPixelwise = nn.L1Loss(reduction='mean')
        self.criterionDis = network.Discriminator(self.args.num_s).to(self.args.device)

        params1 = []
        params1 += list(self.net_MSI2S.parameters())
        params1 += list(self.net_s2img.parameters())
        params1 += list(self.net_LR2s.parameters())
        params2 = []
        params2 += list(self.net_PSF.parameters())
        params2 += list(self.net_G_HR2MSI.parameters())
        self.optimizer1 = optim.Adam(params1, lr=self.args.lr)
        self.optimizer2 = optim.Adam(params2, lr=self.args.lr * 0.2)
        self.scheduler1 = lr_scheduler.LambdaLR(self.optimizer1, lr_lambda=lambda_rule)
        self.scheduler2 = lr_scheduler.LambdaLR(self.optimizer2, lr_lambda=lambda_rule)

    def train(self):

        for epoch in range(1, self.args.niter + self.args.niter_decay + 1):
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            # first lr process
            # self.rec_lr_s  = self.net_LR2s(self.real_lhsi)[0]
            self.rec_lr_s = self.net_LR2s(self.lr_hsi)
            self.rec_lr_lr = self.net_s2img(self.rec_lr_s)

            # second msi process
            self.rec_msi_s, self.msi_out1, self.msi_out2 = self.net_MSI2S(self.hr_msi)
            self.rec_msi_lrs = self.net_PSF(self.rec_msi_s)
            self.rec_RecoveredAbundance = self.net_Spacetransform(
                self.rec_lr_s.permute(0, 2, 3, 1), (self.rec_lr_s + self.rec_msi_lrs).permute(0, 2, 3, 1)
            )
            # self.rec_msi= self.net_m2img(self.rec_msi_s)
            self.rec_lr_lr = self.net_s2img(self.rec_RecoveredAbundance.permute(0, 3, 1, 2))
            self.rec_msi_hr = self.net_s2img(self.rec_msi_s)
            # third PSF and SRF
            self.rec_msi_msi = self.net_G_HR2MSI(self.rec_msi_hr)

            #loss
            self.loss_lr_pixelwise = self.criterionPixelwise(self.lr_hsi, self.rec_lr_lr) * self.args.lambda_A
            self.loss_lr_s_sumtoone = self.criterionSumToOne(self.rec_lr_s) * self.args.lambda_D
            self.loss_lr_sparse = self.criterionSparse(self.rec_lr_s) * self.args.lambda_F

            # self.loss_hrmsi_pixelwise = self.criterionPixelwise(self.real_hmsi, self.rec_msi) * self.opt.lambda_A
            self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_s_sumtoone + self.loss_lr_sparse
            # MSI
            self.loss_msi_pixelwise = self.criterionPixelwise(self.hr_msi,
                                                              self.rec_msi_msi) * self.args.lambda_B  # SRF
            self.loss_msi_s_sumtoone = self.criterionSumToOne(self.rec_msi_s) * self.args.lambda_D
            self.loss_msi_sparse = self.criterionSparse(self.rec_msi_s) * self.args.lambda_D
            self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_s_sumtoone + self.loss_msi_sparse
            # PSF
            self.loss_msi_ss_lr = self.criterionPixelwise(self.rec_lr_s, self.rec_msi_lrs) * self.args.lambda_B
            # !!
            # self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_lrhsi_lrmsi, self.rec_hrmsi_lrmsi) * self.opt.lambda_F
            self.lossC = self.criterionDis(self.msi_out1, self.msi_out2, self.rec_lr_s) * self.args.lambda_F
            self.loss = self.loss_lr + self.loss_msi + self.loss_msi_ss_lr + self.lossC
            self.loss.backward()
            self.optimizer1.step()
            self.scheduler1.step()
            self.optimizer2.step()
            self.scheduler2.step()
            if epoch % self.args.freq == 0:
                with torch.no_grad():
                    print("____________________________________________")
                    print('epoch:{} lr1:{} lr2:{} 保存结果'.format(epoch, self.optimizer1.param_groups[0]['lr'],
                                                                   self.optimizer1.param_groups[0]['lr']))
                    print('************')

                scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out.mat'),
                                 {'Out': self.rec_msi_hr.data.cpu().numpy()[0].transpose(1, 2, 0)})

        return self.rec_msi_hr.data.cpu().numpy()[0].transpose(1, 2, 0)


if __name__ == "__main__":
    pass