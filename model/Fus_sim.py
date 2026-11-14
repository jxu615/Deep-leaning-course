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
from .dataset import Dataset
from . import network

class fus(Dataset):
    def __init__(self, args, srf,psf):
        super().__init__(args)
        self.gt = self.tensor_gt
        self.args = args
        self.hs_band = self.tensor_lr_hsi.shape[1]
        self.ms_band = self.tensor_hr_msi.shape[1]
        self.h = self.tensor_hr_msi.shape[2]
        self.l_h = self.tensor_lr_hsi.shape[2]
        self.n_r = self.h // self.args.window
        psf_est = np.reshape(psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor)) #1 1 ratio ratio 大小的tensor
        self.psf = torch.tensor(psf_est).to(self.args.device).float()
        srf_est = np.reshape(srf.T, newshape=(srf.shape[1], srf.shape[0], 1, 1)) #self.srf.T 有一个T转置 (8, 191, 1, 1)
        self.srf = torch.tensor(srf_est).to(self.args.device).float()
        self.lr_hsi = self.tensor_lr_hsi
        self.hr_msi = self.tensor_hr_msi

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - self.args.niter) / float(self.args.niter_decay + 1)
            return lr_l


        self.net_LR2s = network.define_img2s(self.ms_band, self.hs_band,self.h,self.args.num_s,self.args.device)
        self.net_s2img = network.define_s2img(input_ch=self.args.num_s, output_ch=self.hs_band,gpu_ids=self.args.device)

        # define psf function
        self.net_PSF = network.PSF_down()
        # reconstruct msi image from target HRHSI image
        self.net_G_HR2MSI = network.SRF_down()

        self.criterionSumToOne = network.SumToOneLoss().to(self.args.device)
        self.criterionSparse = network.SparseKLloss().to(self.args.device)
        self.criterionPixelwise = nn.L1Loss(reduction='mean')

        params2 =self.net_s2img.parameters()
        params3 =self.net_LR2s.parameters()


        self.optimizer2 = optim.Adam(params2, lr=self.args.lr)
        self.scheduler2 = lr_scheduler.LambdaLR(self.optimizer2, lr_lambda=lambda_rule)

        self.optimizer3 = optim.Adam(params3, lr=self.args.lr)
        self.scheduler3 = lr_scheduler.LambdaLR(self.optimizer3, lr_lambda=lambda_rule)

    def train(self):
        flag_best = [10, 0, 'data']
        for epoch in range(1, self.args.niter + self.args.niter_decay + 1):
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()
            # first lr process
            # self.rec_lr_s  = self.net_LR2s(self.real_lhsi)[0]
            self.Ab_hsi,self.Ab_hsiup,self.Ab_msi,self.z = self.net_LR2s(self.hr_msi,self.lr_hsi)
            self.rec_lr_lr = self.net_s2img(self.Ab_hsi)

            # self.rec_msi= self.net_m2img(self.rec_msi_s)
            self.rec_msi_hr = self.net_s2img(self.z)
            # third PSF and SRF
            self.rec_msi_msi = self.net_G_HR2MSI(self.rec_msi_hr,self.srf)
            self.rec_msi_lrs = self.net_PSF(self.rec_msi_hr,self.psf,self.args.scale_factor)

            # loss rec
            self.loss_lr_pixelwise = self.criterionPixelwise(self.lr_hsi, self.rec_lr_lr) * self.args.lambda_A
            # loss sto sp
            self.loss_lr_s_sumtoone = self.criterionSumToOne(self.Ab_hsi) * self.args.lambda_D
            self.loss_lr_sparse = self.criterionSparse(self.Ab_hsi) * self.args.lambda_F
            self.loss_msi_s_sumtoone = self.criterionSumToOne(self.z) * self.args.lambda_D
            self.loss_msi_sparse = self.criterionSparse(self.z) * self.args.lambda_D
            self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_s_sumtoone + self.loss_lr_sparse
            #loss deg
            self.loss_msi_ss_lr = self.criterionPixelwise(self.lr_hsi, self.rec_msi_lrs) * self.args.lambda_B
            self.loss_msi_pixelwise = self.criterionPixelwise(self.hr_msi,self.rec_msi_msi) * self.args.lambda_B  # SRF
            self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_s_sumtoone + self.loss_msi_sparse
            # PSF
            # !!
            # self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_lrhsi_lrmsi, self.rec_hrmsi_lrmsi) * self.opt.lambda_F
            self.loss = self.loss_lr + self.loss_msi + self.loss_msi_ss_lr
            self.loss.backward()
            self.optimizer2.step()
            self.scheduler2.step()
            self.optimizer3.step()
            self.scheduler3.step()
            if epoch % self.args.freq == 0:
                with torch.no_grad():
                    print("____________________________________________")
                    print('epoch:{} lr1:{} lr2:{} 保存结果'.format(epoch, self.optimizer2.param_groups[0]['lr'],
                                                                   self.optimizer2.param_groups[0]['lr']))
                    print('************')
                    hrhsi_est_numpy = self.rec_msi_hr.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    gt_numpy = self.gt.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(gt_numpy, hrhsi_est_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(gt_numpy - hrhsi_est_numpy))
                    metr = "precision:\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(metr)
                    file_name = os.path.join(self.args.expr_dir, 'precision.txt')
                    with open(file_name, 'a') as opt_file:
                        opt_file.write('epoch:{}'.format(epoch))
                        opt_file.write('\n')
                        opt_file.write(metr)
                        opt_file.write('\n')
                        opt_file.write('\n')
                    print('************')
                    if sam < flag_best[0] and psnr > flag_best[1]:
                        flag_best[0] = sam
                        flag_best[1] = psnr
                        flag_best[2] = self.rec_msi_hr  # 保存四维tensor

        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out.mat'),{'Out': flag_best[2].data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'hsiup_ab.mat'),
                         {'Out': self.Ab_hsiup.data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'msi_ab.mat'),
                         {'Out': self.Ab_msi.data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'out_ab.mat'),
                         {'Out': self.z.data.cpu().numpy()[0].transpose(1, 2, 0)})

        scipy.io.savemat(os.path.join(self.args.expr_dir, 'endmember.mat'),
                         {'Out': self.net_s2img.net[0].weight.data.squeeze(-1).squeeze(-1).cpu().numpy()})

        file_name = os.path.join(self.args.expr_dir, 'precision.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('epoch:{}'.format(epoch))
            opt_file.write('\n')
            opt_file.write(metr)
            opt_file.write('\n')
            opt_file.write('\n')
        return flag_best[2].data.cpu().numpy()[0].transpose(1, 2, 0)


if __name__ == "__main__":
    pass