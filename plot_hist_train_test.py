#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:05:35 2024
plot distribution of training and testing data
@author: g260218
"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorResNet
from datasets import myDataset
from funs_prepost import var_denormalize,nc_load_all,ntpd,plot_mod_vs_obs,plot_distri

import torch
import pandas as pd

import sys
import importlib
mod_name= 'par11'          #'par04' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 

import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    suf0 = mod_para.suf0
    rtra = mod_para.rtra
    # rtra = 0.95
    ivar_lr = mod_para.ivar_lr
    ivar_hr = mod_para.ivar_hr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(ivar_lr)
    nchl_o = len(ivar_hr)
    
    nrep = mod_para.nrep
    rep = list(range(0,nrep))
    rep = [0]
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    epoc_num = [50,100]
    # epoc_num = np.arange(40,opt.N_epochs+1)
    key_ep_sort = 1 # to use epoc here or load sorted epoc no. 
    nepoc = 2 # no. of sorted epochs for analysis

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    train_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='train',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr) #,ind_sort=ind_sort

    test_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='val',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr) # rtra = rtra,
    
    opath_st = 'statistics' + suf +'/'
    if not os.path.exists(opath_st):
        os.makedirs(opath_st)

    data_train = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )
    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=opt.n_cpu,
    )        
    
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    # get logitude and latitude of data 
    nc_f = test_set.files_hr[0]
    lon = nc_load_all(nc_f,0)[1]
    lat = nc_load_all(nc_f,0)[2]
    mask = nc_load_all(nc_f,0)[10] # original data
    
    in_path = 'results_test/'+'SRF_'+str(opt.up_factor)+'/' # hr_all, etc
    
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)']
    
    out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_st'+'/'
    os.makedirs(out_path, exist_ok=True)
        
    # get all training data 
    lr_all_train = []
    hr_all_train = []
    for i, dat in enumerate(data_train):                
        dat_lr = Variable(dat["lr"].type(Tensor))
        dat_hr = Variable(dat["hr"].type(Tensor))
        hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
        lr_norm0 = var_denormalize(dat_lr.detach().cpu().numpy(),varm_hr)
        
        # get mask for time step, one batch may cover muliple files
        # for ib in range(0,opt.batch_size):
        ib = int(opt.batch_size/2) # use mask of the middle sample to represent the mask of this batch
        it = i*opt.batch_size + ib  # it is no. of time steps in dataset, not true time
        indf = int(it/ntpd)  # id of ncfile that contain field with id index
        indt = it % ntpd  # the i th time step in a ncfile
        
        nc_f = train_set.files_hr[indf]
        mask = nc_load_all(nc_f,indt)[10] # mask in a batch        
        hr_norm0[:,:,mask] = np.nan
        hr_all_train.append(hr_norm0)
        
        nc_f = train_set.files_lr[indf]
        mask = nc_load_all(nc_f,indt)[10] # mask in a batch        
        lr_norm0[:,:,mask] = np.nan
        lr_all_train.append(lr_norm0)

    hr_all_train = np.array(hr_all_train).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
    lr_all_train = np.array(lr_all_train).reshape(-1,nchl,int(hr_shape[0]/opt.up_factor),int(hr_shape[1]/opt.up_factor)) # [Nt,c,H,W]
    
    # get lr test data 
    lr_all = []
    hr_all = []
    for i, dat in enumerate(data_test):                
        dat_lr = Variable(dat["lr"].type(Tensor))
        dat_hr = Variable(dat["hr"].type(Tensor))
        hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
        lr_norm0 = var_denormalize(dat_lr.detach().cpu().numpy(),varm_hr)
        
        # get mask for time step, one batch may cover muliple files
        # for ib in range(0,opt.batch_size):
        ib = int(opt.batch_size/2) # use mask of the middle sample to represent the mask of this batch
        it = i*opt.batch_size + ib  # it is no. of time steps in dataset, not true time
        indf = int(it/ntpd)  # id of ncfile that contain field with id index
        indt = it % ntpd  # the i th time step in a ncfile
        
        nc_f = test_set.files_hr[indf]
        mask = nc_load_all(nc_f,indt)[10] # mask in a batch        
        hr_norm0[:,:,mask] = np.nan
        hr_all.append(hr_norm0)
        
        nc_f = test_set.files_lr[indf]
        mask = nc_load_all(nc_f,indt)[10] # mask in a batch        
        lr_norm0[:,:,mask] = np.nan
        lr_all.append(lr_norm0)

    hr_all = np.array(hr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
    lr_all = np.array(lr_all).reshape(-1,nchl,int(hr_shape[0]/opt.up_factor),int(hr_shape[1]/opt.up_factor)) # [Nt,c,H,W]
    
    # show distribution of training/testing dataset
    for i in range(nchl):
        ichl = ivar_hr[i]-3
        # load saved reconstruced hr (using cal_metrics_intp.py)
        # filename = in_path + 'c%d_'%(ichl)+'hr_all'+'_train%4.2f'%(rtra)+'.npz'
        # datald = np.load(filename) # load
        # hr_all = datald['v0']
        unit_var = unit_suv[ichl]
        
        # plot distribution of reconstructed vs target, all data, histogram
        axlab = (unit_var,'Frequency','')
        leg = ['hr_train','lr_train','hr_test','lr_test'] #,'nearest'
        var1 = hr_all_train[:,i,:,:].flatten()
        var2 = lr_all_train[:,i,:,:].flatten()
        var3 = hr_all[:,i,:,:].flatten()
        var4 = lr_all[:,i,:,:].flatten()
        var = [var1[~np.isnan(var1)],var2[~np.isnan(var2)],
               var3[~np.isnan(var3)],var4[~np.isnan(var4)],
               ]
        figname = out_path+"c%d" % (ichl) +'_dist_train_test'+'.png'
        plot_distri(var,figname,bins=20,axlab=axlab,leg=leg,
                       figsize=(10, 5), fontsize=16,capt='(a)')
    
    