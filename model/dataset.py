#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class Dataset
    Generate simulation data
~~~~~~~~~~~~~~~~~~~~~~~~
Function:
    downsamplePSF: The function of this function is to ensure that the same Gaussian downsampling method is used with matlab.


"""

import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
import random
import xlrd

class Dataset():
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.args = args
        self.srf_gt = self.get_spectral_response(self.args.data_name)

        self.psf_gt = self.matlab_style_gauss2D(shape=(self.args.scale_factor, self.args.scale_factor),
                                                sigma=self.args.sigma)
        self.sp_range = self.get_sp_range(self.srf_gt)
        data_folder = os.path.join(self.args.default_datapath, args.data_name)

        if self.args.simulate == 1 :
            if os.path.exists(data_folder):

                data_path = os.path.join(data_folder, "REF.mat")
            else:
                return 0
            self.gt = io.loadmat(data_path)['REF']

            self.lr_hsi = self.generate_low_HSI(self.gt, self.args.scale_factor)

            self.hr_msi = self.generate_MSI(self.gt, self.srf_gt)

            self.lr_msi_fmsi = self.generate_low_HSI(self.hr_msi, self.args.scale_factor)


            self.lr_msi_fhsi = self.generate_MSI(self.lr_hsi, self.srf_gt)

            self.tensor_gt = torch.from_numpy(self.gt.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(args.device)
            self.tensor_lr_hsi = torch.from_numpy(self.lr_hsi.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(
                args.device)
            self.tensor_hr_msi = torch.from_numpy(self.hr_msi.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(
                args.device)
            self.tensor_lr_msi_fmsi = torch.from_numpy(self.lr_msi_fmsi.transpose(2, 0, 1).copy()).unsqueeze(
                0).float().to(
                args.device)
            self.tensor_lr_msi_fhsi = torch.from_numpy(self.lr_msi_fhsi.transpose(2, 0, 1).copy()).unsqueeze(
                0).float().to(
                args.device)

            self.print_options()

            self.save_psf_srf()

            self.save_lrhsi_hrmsi()
            print("readdata over")

        else:
            if os.path.exists(data_folder):

                data_path = os.path.join(data_folder, "*.mat")
            else:
                return 0
            self.lr_hsi = io.loadmat(data_path)['lrhsi']
            self.hr_msi = io.loadmat(data_path)['hrmsi']

            self.tensor_lr_hsi = torch.from_numpy(self.lr_hsi.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(
                args.device)
            self.tensor_hr_msi = torch.from_numpy(self.hr_msi.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(
                args.device)

            self.print_options()

            self.save_psf_srf()

            self.save_lrhsi_hrmsi()
            print("readdata over")





    def matlab_style_gauss2D(self, shape=(3, 3), sigma=2):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_spectral_response(self, data_name):
        xls_path = os.path.join(self.args.sp_root_path, data_name + '.xls')
        # xls_path = os.path.join(r'E:\Code\coupled\data\spectral_response', data_name + '.xls')
        if not os.path.exists(xls_path):
            raise Exception("spectral response path does not exist")
        data = xlrd.open_workbook(xls_path)
        table = data.sheets()[0]

        num_cols = table.ncols
        cols_list = [np.array(table.col_values(i)).reshape(-1, 1) for i in range(0, num_cols)]

        sp_data = np.concatenate(cols_list, axis=1)
        sp_data = sp_data / (sp_data.sum(axis=0))

        return sp_data

    def get_sp_range(self, srf_gt):
        HSI_bands, MSI_bands = srf_gt.shape

        assert (HSI_bands > MSI_bands)
        sp_range = np.zeros([MSI_bands, 2])
        for i in range(0, MSI_bands):
            index_dim_0, index_dim_1 = np.where(srf_gt[:, i].reshape(-1, 1) > 0)
            sp_range[i, 0] = index_dim_0[0]
            sp_range[i, 1] = index_dim_0[-1]
        return sp_range

    def downsamplePSF(self, img, sigma, stride):
        def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
            m, n = [(ss - 1.) / 2. for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride, stride), sigma)
        if img.ndim == 3:
            img_w, img_h, img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w, img_h = img.shape
            img = img.reshape((img_w, img_h, 1))
        from scipy import signal
        out_img = np.zeros((img_w // (stride), img_h // (stride), img_c))
        for i in range(img_c):
            out = signal.convolve2d(img[:, :, i], h, 'valid')
            out_img[:, :, i] = out[::stride, ::stride]
        return out_img

    def generate_low_HSI(self, img, scale_factor):
        (h, w, c) = img.shape
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        return img_lr

    def generate_MSI(self, img, srf_gt):
        w, h, c = img.shape
        self.msi_channels = srf_gt.shape[1]
        print(srf_gt.shape)
        print(img.shape)
        if srf_gt.shape[0] == c:
            img_msi = np.dot(img.reshape(w * h, c), srf_gt).reshape(w, h, srf_gt.shape[1])
        else:
            raise Exception("The shape of sp matrix doesnot match the image")
        return img_msi

    def print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        if not os.path.exists(self.args.expr_dir):
            os.makedirs(self.args.expr_dir)

        file_name = os.path.join(self.args.expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def save_psf_srf(self):
        psf_name = os.path.join(self.args.expr_dir, 'psf_gt.mat')
        srf_name = os.path.join(self.args.expr_dir, 'srf_gt.mat')
        io.savemat(psf_name, {'psf_gt': self.psf_gt})
        io.savemat(srf_name, {'srf_gt': self.srf_gt})

    def save_lrhsi_hrmsi(self):
        lr_hsi_name = os.path.join(self.args.expr_dir, 'lr_hsi.mat')
        hr_msi_name = os.path.join(self.args.expr_dir, 'hr_msi.mat')
        io.savemat(lr_hsi_name, {'lr_hsi': self.lr_hsi})
        io.savemat(hr_msi_name, {'hr_msi': self.hr_msi})
        io.savemat(os.path.join(self.args.expr_dir, 'gt_lr_msi.mat'),
                   {'lr_msi_fhsi_gt': self.lr_msi_fhsi, 'lr_msi_fmsi_gt': self.lr_msi_fmsi})
