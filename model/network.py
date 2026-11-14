import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import os

import torch.fft
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def init_net(net,device, init_type, init_gain,initializer):
    print(device)
    net.to(device)  #gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer :

        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net


class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))

        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        # print(input[0,:,:])
        loss = self.loss(input, target_tensor)
        # loss = torch.sum(torch.abs(target_tensor - input))
        return loss


def kl_divergence(p, q):
    p = F.softmax(p,dim=1)
    q = F.softmax(q,dim=1)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self, input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


class hsi(nn.Module):
    def __init__(self,ch,output_ch):
        super(hsi, self).__init__()
        self.Conv_in = nn.Conv2d(in_channels=ch, out_channels=output_ch , kernel_size=1, stride=1, padding=0,bias=False)
        self.Conv_1 = nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)
        self.Conv_2 = nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x1 = self.relu(self.Conv_in(x))
        x2 = self.relu(self.Conv_1(x1))
        out = self.relu(self.Conv_2(x1+x2))
        return out

class FFT(nn.Module):
    def __init__(self,input_ch):
        super(FFT, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=input_ch, out_channels=input_ch, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=input_ch, out_channels=input_ch, kernel_size=3, stride=1,padding=1, bias=False)
        self.conv_fft_r = nn.Conv2d(in_channels=input_ch, out_channels=input_ch, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_fft_i = nn.Conv2d(in_channels=input_ch, out_channels=input_ch, kernel_size=1, stride=1,padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=input_ch, kernel_size=3, stride=1,padding=1, bias=False),
        )
        self.BN = nn.BatchNorm2d(input_ch)

    def forward(self,x,y):
        fft_x = torch.fft.fft2(x)
        real_x = self.BN(self.relu(self.conv_fft_r(fft_x.real)))
        imag_x = self.BN(self.relu(self.conv_fft_i(fft_x.imag)))
        fft_x = torch.complex(real_x, imag_x)
        fft_y = torch.fft.fft2(y)
        real_y = self.BN(self.relu(self.conv_fft_r(fft_y.real)))
        imag_y = self.BN(self.relu(self.conv_fft_i(fft_y.imag)))
        fft_y = torch.complex(real_y, imag_y)
        x_p = torch.angle(fft_x)
        x_m = torch.abs(fft_x)
        y_m = torch.abs(fft_y)
        mask1 = self.mask(y_m)
        m = self.mask(x_m)
        p = x_p
        fft_rec = m * (torch.cos(p) + 1j * torch.sin(p))
        z = torch.complex(fft_rec.real,fft_rec.imag)
        z = torch.abs(torch.fft.ifft2(z))
        #z = torch.abs(fft_x)
        fft_y = torch.abs(torch.fft.ifft2(fft_y))
        return z,fft_y

class img2Delta(nn.Module):
    def __init__(self,msi_ch,hsi_ch,h,output_ch):
        super(img2Delta, self).__init__()
        self.h = h
        self.FFT = FFT(output_ch)
        self.MSC = MSC(msi_ch,output_ch)
        self.hsi = hsi(hsi_ch,output_ch)
        self.ex = EX(output_ch)
        self.conv_1 = nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.conv_3 = nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self,x,y):
        x = self.MSC(x)
        y = self.hsi(y)
        xout  = self.ex(x)
        yout = self.ex(y)
        yupout = F.interpolate(y, size=(self.h, self.h), mode='bilinear')
        #xout = self.softmax(xout)
        #yupout = self.softmax(yupout)
        yout = self.softmax(yout)
        z,y_2 = self.FFT(xout,yupout)
        y_out = self.relu(self.conv_1(yupout+y_2))
        z_out = self.relu(self.conv_2(z+xout))
        out = self.softmax(self.relu(self.conv_3(y_out+z_out)))
        #y_2 = self.softmax(y_2)
        yupout = self.softmax(yupout)
        return yout, yupout, xout, out

def define_img2s(msi_ch,his_ch,h,output_ch,gpu_ids, init_type='kaiming', init_gain=0.02,initializer=True):
    net = img2Delta(msi_ch,his_ch,h,output_ch)
    return init_net(net, gpu_ids, init_type, init_gain,initializer)




class EX(nn.Module):
    def __init__(self,ch):
        super(EX, self).__init__()
        self.Conv_in = nn.Conv2d(in_channels=ch, out_channels=ch , kernel_size=1, stride=1, padding=0,bias=False)
        self.Conv_1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)
        self.Conv_2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x1 = self.relu(self.Conv_in(x))
        x2 = self.relu(self.Conv_1(x1))
        out = self.relu(self.Conv_2(x1+x2))
        return out







def define_s2img(input_ch, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02,initializer=True):
    net = S2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, gpu_ids, init_type, init_gain,initializer)


class S2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(S2Img, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.net(x)






class MSC(nn.Module):
    def __init__(self,input_ch,output_ch):
        super(MSC, self).__init__()
        self.conv_5 = nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv_3 = nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=5, stride=1, padding=2,bias=False)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self,x):
        co3 = self.relu(self.conv_3(x))
        co5 = self.relu(self.conv_5(x))
        out = torch.add(co3,co5)
        return out






class PSF_down():

    def __call__(self, input_tensor, psf, ratio): #PSF为#1 1 ratio ratio 大小的tensor
        _,C,_,_=input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3]
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1) #8X1X8X8
                                               #input_tensor: 1X8X400X400
        output_tensor = F.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=C) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor


class SRF_down():
    def __call__(self, input_tensor, srf): # srf 为 ms_band hs_bands 1 1 的tensor
            output_tensor = F.conv2d(input_tensor, srf, None)
            return output_tensor

def define_spatial_transform(use_gpu=True):
    net = SpatialTransformation(use_gpu)
    # net = SpatialTransformer(img_size)
    return net

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(
            torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0),
        )
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]),)

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu == True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):
        im = F.pad(im, (0, 0, 1, 1, 1, 1, 0, 0))
        batch_size, height, width, channels = im.shape
        batch_size, out_height, out_width = x.shape
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        x = x + 1
        y = y + 1
        max_x = width - 1
        max_y = height - 1
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        dim2 = width
        dim1 = width * height
        base = self.repeat(torch.arange(0, batch_size) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1, 0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1, 0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1, 0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1, 0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1, 0).expand(dim, channels))

        x1_f = x1.float()
        y1_f = y1.float()
        dx = x1_f - x
        dy = y1_f - y
        wa = (dx * dy).transpose(1, 0)
        wb = (dx * (1 - dy)).transpose(1, 0)
        wc = ((1 - dx) * dy).transpose(1, 0)
        wd = ((1 - dx) * (1 - dy)).transpose(1, 0)
        output = torch.sum(torch.squeeze(torch.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]
        batch_size, height, width = dx.shape
        x_mesh, y_mesh = self.meshgrid(height, width)
        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        return self.interpolate(moving_image, x_new, y_new)
class NonZeroClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1e8)


class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1)


class SumToOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0, 10)
                w.div_(w.sum(dim=1, keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0, 5)
