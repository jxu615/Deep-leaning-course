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
        data_folder = os.path.join(self.args.default_datapath, args.data_name)
        if os.path.exists(data_folder):

            data_path = os.path.join(data_folder, "REF.mat")
            print("datapath=",'data_path')
        else:
            return 0


        self.lr_hsi = io.loadmat(data_path)['lrhsi']
        self.hr_msi = io.loadmat(data_path)['hrmsi']

        self.tensor_lr_hsi = torch.from_numpy(self.lr_hsi.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(
                args.device)
        self.tensor_hr_msi = torch.from_numpy(self.hr_msi.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(
                args.device)

        print("readdata over")

