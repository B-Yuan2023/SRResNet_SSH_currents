import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from funs_prepost import nc_load_all,nc_var_normalize

rval = 0.1 # ratio of validation dataset
ntpd = 24 # number of time steps in an nc file

def lr_transform(hr_height, hr_width, up_factor):
    return transforms.Compose(
        [
            transforms.Resize((hr_height // up_factor, hr_width // up_factor), Image.BICUBIC), # transforms.InterpolationMode.BICUBIC
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ]
    )

def hr_transform(hr_height,hr_width):
    return transforms.Compose(
        [
            transforms.Resize((hr_height, hr_width), Image.BICUBIC), # 
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ]
    )

def find_maxmin_global(files, ivar=[3,3,3]):
    # files = sorted(glob.glob(dirname + "/*.nc"))
    # nfile = len(files)
    # files = files[:int(nfile*rtra)]
    file_varm = [] 
    ind_varm = np.ones((len(ivar),2),dtype= np.int64)
    varmaxmin = np.ones((len(ivar),2))
    varmaxmin[:,0] *= -10e6 # maximum 
    varmaxmin[:,1] *= 10e6 # minimum 
    for i in range(len(ivar)):
        for indf in range(len(files)):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            if varmaxmin[i,0]<var.max():
                varmaxmin[i,0] = var.max()
                ind_varm[i,0] = np.argmax(var)
                file_max = nc_f
            if varmaxmin[i,1]>var.min():
                varmaxmin[i,1] = var.min()
                ind_varm[i,1] = np.argmin(var)
                file_min = nc_f
            # varmaxmin[i,0] = max(varmaxmin[i,0],var.max())
            # varmaxmin[i,1] = min(varmaxmin[i,1],var.min())
        file_varm.append([file_max,file_min])
    return varmaxmin,ind_varm,file_varm


def my_loss(output,target,nlm=2):  
    # output, target: dimensionless tensor[N,C,H,W]
    # nlm: key for loss function, originally used as norm order
    if nlm == 1:    # norm1
        loss = torch.norm(output-target,1)
    elif nlm == 2:      # norm2
        loss = torch.norm(output-target,2)
    elif nlm == 3:      # norm3
        loss = torch.norm(output-target,3)
    elif nlm == 4:      # mae
        loss = torch.mean(torch.abs(output - target))
    elif nlm == 5:      # mse
        loss = torch.mean((output - target)**2)
    return loss

class myDataset(Dataset):
    def __init__(self, dir_lr,dir_hr, hr_shape,up_factor=4,mode='train', rtra=0.7,
                 ivar_lr=[3,],ivar_hr=[3,],varm_lr=None,varm_hr=None,ind_sort=None):
        hr_height, hr_width = hr_shape
        self.mode = mode
        self.ivar_lr = ivar_lr
        self.ivar_hr = ivar_hr
        self.varm_lr = varm_lr
        self.varm_hr = varm_hr
        # Transforms for low resolution images and high resolution images
        self.lr_transform = lr_transform(hr_height, hr_width, up_factor)
        self.hr_transform = hr_transform(hr_height,hr_width)
        
        self.files_lr = sorted(glob.glob(dir_lr + "/*.nc"))
        self.files_hr = sorted(glob.glob(dir_hr + "/*.nc"))
        assert len(self.files_lr) == len(self.files_hr),'lr & hr samples not match!'
        nfile = len(self.files_lr)
        if ind_sort is None: # default file order
            # ind_train = np.arange(15,int(nfile*rtra)+15)    # exclude max, only for schism
            ind_train = np.arange(0,int(nfile*rtra))    # exclude max
            ind_valid= np.delete(np.arange(0,nfile),ind_train)
        else:   # 20240320: file order with one var from small to large (hr)
            ind_train = np.arange(0,int(nfile*rtra))    # 
            ind_valid= np.delete(np.arange(0,nfile),ind_train)
            self.files_lr = [self.files_lr[i] for i in ind_sort] 
            self.files_hr = [self.files_hr[i] for i in ind_sort]

        if mode == "train":
            self.files_lr = [self.files_lr[i] for i in ind_train] 
            self.files_hr = [self.files_hr[i] for i in ind_train] # list can not directly use array as index 
        elif mode == "val":
            self.files_lr = [self.files_lr[i] for i in ind_valid] 
            self.files_hr = [self.files_hr[i] for i in ind_valid]
        else:
            self.files_lr = self.files_lr[int(nfile*(rtra+rval)):]
            self.files_hr = self.files_hr[int(nfile*(rtra+rval)):]


    def __getitem__(self, index):
        indf = int(index/ntpd)  # id of ncfile that contain field with id index
        indt = index % ntpd  # the i th time step in a ncfile

        nc_f = self.files_hr[indf % len(self.files_hr)]
        data = nc_var_normalize(nc_f,indt,self.ivar_hr,self.varm_hr)        
        x = []
        for i in range(data.shape[2]): # data(H,W,C)
            img = Image.fromarray((data[:,:,i])) # single channel to img
            x.append(self.hr_transform(img))
        img_hr = torch.cat(x)
        
        nc_f = self.files_lr[indf % len(self.files_lr)]
        data = nc_var_normalize(nc_f,indt,self.ivar_lr,self.varm_lr)        
        x = []
        for i in range(data.shape[2]): # data(H,W,C)
            img = Image.fromarray((data[:,:,i])) # single channel to img
            x.append(self.lr_transform(img))
        img_lr = torch.cat(x)
        
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files_lr)*ntpd
