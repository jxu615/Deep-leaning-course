
import torch
import torch.nn as nn
import time
import numpy as np
import hues
import os
import random
import scipy.io as sio
from model.config import args
from model.evaluation import MetricsCal
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"!!!!!!!!!!Please contact me if you have problems running the code： JinXu0615@163.com!!!!!!!!!!!!"

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
setup_seed(30)

if args.simulate == 0 :
    from model.srf_layer_real import Blind
    blind = Blind(args)
    start = time.perf_counter()
    lr_msi_fhsi_est, lr_msi_fmsi_est = blind.train()
    blind.get_save_result()
    psf = blind.model.psf.data.cpu().detach().numpy()[0, 0, :, :]
    srf = blind.model.srf.data.cpu().detach().numpy()[:, :, 0, 0].T
    end = time.perf_counter()
    elapsed_S1 = end - start
    # from model.Fus_real import fus
    # MoiT = fus(args,srf,psf)
    # start = time.perf_counter()
    # HrHSI = MoiT.train()
    # end = time.perf_counter()
    # elapsed_S2 = end - start
    # print('times:',elapsed_S1, elapsed_S2)

if args.simulate == 1 :
    from model.srf_layer_sim1 import Blind
    blind = Blind(args)
    start = time.perf_counter()
    lr_msi_fhsi_est, lr_msi_fmsi_est = blind.train()
    blind.get_save_result()
    psf = blind.model.psf.data.cpu().detach().numpy()[0, 0, :, :]
    srf = blind.model1.srf.data.cpu().detach().numpy()[:, :, 0, 0].T
    psf_gt = blind.psf_gt
    srf_gt = blind.srf_gt
    psf_res = np.sum(np.abs(psf_gt - psf))
    srf_res = np.sum(np.abs(srf_gt - srf))
    print("psf与实际值的差异为：", psf_res)
    print("srf与实际值的差异为：", srf_res)
    end = time.perf_counter()
    elapsed_S1 = end - start
    from model.Fus_sim import fus
    MoiT = fus(args,srf,psf_gt)
    start = time.perf_counter()
    HrHSI = MoiT.train()
    end = time.perf_counter()
    elapsed_S2 = end - start
    print("psf与实际值的差异为：", psf_res)
    print("srf与实际值的差异为：", srf_res)
    print('times:', elapsed_S1, elapsed_S2)






