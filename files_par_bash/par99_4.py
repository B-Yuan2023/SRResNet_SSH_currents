#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:16:41 2024
parameters 

@author: g260218
"""
import sys
import argparse
import numpy as np
# from datasets import find_maxmin_global
# import glob

# def create_parser():
parser = argparse.ArgumentParser()
parser.add_argument("--n0_epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--N_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dir_lr", type=str, default="../out2d_32", help="directory of lr dataset")
parser.add_argument("--dir_hr", type=str, default="../out2d_128", help="directory of hr dataset")
parser.add_argument("--up_factor", type=int, default=4, help="upscale factor") # match with dirs
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate") # default 0.001. 0.0002
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient") # default 0.9. 0.5
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
# parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_epoch", type=int, default=10, help="epoch interval between saving image")
parser.add_argument("--sample_interval", type=int, default=200, help="batch interval between saving image")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in the generator")
parser.add_argument("--rlpxl", type=float, default=1.0, help="ratio pixel loss") 
parser.add_argument("--rladv", type=float, default=0.001, help="ratio adversal loss") 
parser.add_argument("--rlper", type=float, default=0.00, help="ratio perception loss") #  0.006
parser.add_argument("--nlm", type=int, default=3, help="norm order") #
opt = parser.parse_args(sys.argv[2:])
print(opt)
    # return opt

krand = 1
nrep = 2

# suf = '_max_var1_nb08'
# suf0 = '_max_var1'

suf = '_res' + str(opt.residual_blocks) + '_max_var1'
suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'

rtra = 0.75 # ratio of training dataset

# ivar_lr = [6,] # default use only ssh [3,3,3]; [3,4,5,6,7,8,9] corresponds to ssh,ud,vd,uw,vw,swh,pwp
# ivar_hr = [8,]
ivar_lr = list(range(3,10)) # default use only ssh [3,3,3]; [3,4,5,6,7,8,9] corresponds to ssh,ud,vd,uw,vw,swh,pwp
ivar_hr = list(range(3,10)) 
# files_lr = sorted(glob.glob(opt.dir_lr + "/*.nc"))
# varm_lr,_,_ = find_maxmin_global(files_lr, ivar_lr)
# files_hr = sorted(glob.glob(opt.dir_hr + "/*.nc"))
# varm_hr = find_maxmin_global(files_hr, ivar_hr)
# estimated based on whole dataset

varm_hr0 = np.array([[ 3.0, -3.0],
       [ 2.1, -2],
       [ 2.1, -2],
       [ 21, -21],
       [ 21, -21],
       [ 7, 0],
       [ 20,0]])
# varm_hr0 = np.array([[  2.80240449,  -2.63572259],
       # [  1.95866261,  -1.65001754],
       # [  2.12618377,  -1.39972807],
       # [ 20.44596922, -13.57177825],
       # [ 16.20933586, -20.8452253 ],
       # [  6.52002267,   0.        ],
       # [ 19.72654343,   0.        ]])
# varm_lr = np.array([[  2.71783665,  -2.62399791],
       # [  1.95866261,  -1.52233336],
       # [  1.07992478,  -0.96283311],
       # [ 20.44596922, -13.57177825],
       # [ 16.14000514, -20.51921248],
       # [  6.4927253 ,   0.03280069],
       # [ 18.22865677,   1.57604408]])
varm_hr = np.zeros((len(ivar_hr),2))
varm_lr = np.zeros((len(ivar_lr),2))
for i in range(len(ivar_hr)):
    if ivar_hr[i] == 3: # ssh 
        varm_hr[i,:] = varm_hr0[0,:]
    elif ivar_hr[i] == 4: # u
        varm_hr[i,:] = varm_hr0[1,:]
    elif ivar_hr[i] == 5: # v
        varm_hr[i,:] = varm_hr0[2,:]
    elif ivar_hr[i] == 6: # uw
        varm_hr[i,:] = varm_hr0[3,:]
    elif ivar_hr[i] == 7: # vw
        varm_hr[i,:] = varm_hr0[4,:]
    elif ivar_hr[i] == 8: # swh
        varm_hr[i,:] = varm_hr0[5,:]
    elif ivar_hr[i] == 9: # pwp
        varm_hr[i,:] = varm_hr0[6,:]
for i in range(len(ivar_lr)):
    if ivar_lr[i] == 3: # ssh 
        varm_lr[i,:] = varm_hr0[0,:]
    elif ivar_lr[i] == 4: # u
        varm_lr[i,:] = varm_hr0[1,:]
    elif ivar_lr[i] == 5: # v
        varm_lr[i,:] = varm_hr0[2,:]
    elif ivar_lr[i] == 6: # uw
        varm_lr[i,:] = varm_hr0[3,:]
    elif ivar_lr[i] == 7: # vw
        varm_lr[i,:] = varm_hr0[4,:]
    elif ivar_lr[i] == 8: # swh
        varm_lr[i,:] = varm_hr0[5,:]
    elif ivar_lr[i] == 9: # pwp
        varm_lr[i,:] = varm_hr0[6,:]
# varm = varm_hr