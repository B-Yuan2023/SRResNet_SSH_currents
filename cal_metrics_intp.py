#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:39:22 2024
calculate metrics for direct interpolation 
@author: g260218
"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import myDataset
from funs_prepost import nc_load_all,var_denormalize,plt_pcolor_list,ntpd
import torch
from pytorch_msssim import ssim as ssim_torch
from math import log10
import pandas as pd

import sys
import importlib
mod_name= 'par99_64_'           # only use 'par99','par99_4','par99_32','par99_64'
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

def cal_metrics(pr,hr,pr_norm0,hr_norm0): # pr,hr are tensors [N,C,H,W], norm0 are arrays,mask
    # hr_norm0[:,:,mask] = np.nan
    nchl = hr.shape[1]
    mse = np.nanmean((pr_norm0 - hr_norm0) ** 2,axis=(0,2,3)) 
    rmse = (mse)**(0.5)
    mae = np.nanmean(abs(pr_norm0 - hr_norm0),axis=(0,2,3))
    
    # to calculate ssim, there should be no nan
    ssim_tor = ssim_torch(pr, hr,data_range=1.0,size_average=False) #.item()  # ,win_size=11
    ssim = np.array([ssim_tor[0,i].item() for i in range(nchl)])
    
    # mask_ud = np.flipud(mask) # dimensionless data flipped
    # hr[:,:,mask_ud.copy()] = np.nan  # for tensor copy is needed. why hr is modified after call
    mse_norm = torch.nanmean(((pr - hr) ** 2).data,axis=(0,2,3)) #.item()
    psnr = np.array([10 * log10(1/mse_norm[i]) for i in range(nchl)]) # for data range in [0,1]
    return rmse, mae, mse, ssim, psnr 

from scipy.interpolate import griddata, interp2d, RBFInterpolator
def interpolate_tensor(tensor, scale_factor, interpolation_function=griddata, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of a tensor using the specified interpolation function.
    
    Parameters:
        tensor (ndarray): Input tensor of shape (N, C, H, W).
        scale_factor (float): Scale factor for the last two dimensions. The new dimensions will be
                              original_dimensions * scale_factor.
        interpolation_function (function, optional): Interpolation function to use.
                                                     Default is griddata from scipy.interpolate.
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').
                                 Default is 'linear'.
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        ndarray: Interpolated tensor of shape (N, C, new_H, new_W).
    """
    N, C, H, W = tensor.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    new_tensor = np.zeros((N, C, new_H, new_W))
    
    # Create 2D grids for interpolation
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Create new 2D grids for interpolated domain
    new_x = np.linspace(0, W - 1, new_W)
    new_y = np.linspace(0, H - 1, new_H)
    new_X, new_Y = np.meshgrid(new_x, new_y)
    
    for n in range(N):
        for c in range(C):
            # Flatten the original grid
            points = np.column_stack((X.flatten(), Y.flatten()))
            values = tensor[n, c].flatten()
            
            # Interpolate using the specified interpolation function
            interpolated = interpolation_function(points, values, (new_X, new_Y), method=method, **kwargs)
            new_tensor[n, c] = interpolated
    new_tensor = torch.from_numpy(new_tensor).type(torch.float)
    return new_tensor

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
    # nrep = mod_para.nrep
    # nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    # epoc_num = [50,100]
    # epoc_num = np.arange(40,opt.N_epochs+1)

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    test_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='val',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,varm_lr=varm_lr,varm_hr=varm_hr) # rtra = rtra,
    
    # opath_st = 'statistics' + suf +'/'
    # if not os.path.exists(opath_st):
    #     os.makedirs(opath_st)
        
    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=opt.n_cpu,
    )        
    Nbatch_t = len(data_test)        
    
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    # get logitude and latitude of data 
    nc_f = test_set.files_hr[0]
    lon = nc_load_all(nc_f,0)[1]
    lat = nc_load_all(nc_f,0)[2]
    # mask = nc_load_all(nc_f,0)[10] # original data
    
    out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+'_mask'+str(kmask)+'/'
    os.makedirs(out_path, exist_ok=True)
    filename99 = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
    filename01 = out_path + 'hr_01per_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
    filename_m = out_path + 'hr_mean_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
    if not os.path.isfile(filename99) or not os.path.isfile(filename01) or not os.path.isfile(filename_m):

        # for direct interpolation
        metrics_re = {'mse_re1': [], 'mae_re1': [], 'rmse_re1': [], 'ssim_re1': [], 'psnr_re1': [],
                      'mse_re2': [], 'mae_re2': [], 'rmse_re2': [], 'ssim_re2': [], 'psnr_re2': [],
                      'mse_re3': [], 'mae_re3': [], 'rmse_re3': [], 'ssim_re3': [], 'psnr_re3': [],
                      }
        metrics_re_bt = {}
        metrics_re_bt_chl = {}
        for key in metrics_re:  # loop only for keys
            metrics_re_bt[key] = []
            metrics_re_bt_chl[key] = []
    
        hr_all = []
        hr_restore1_all = []
        hr_restore2_all = []
        hr_restore3_all = []
        for i, dat in enumerate(data_test):
            dat_lr = Variable(dat["lr"].type(Tensor))
            dat_hr = Variable(dat["hr"].type(Tensor))
            hr_norm0 = var_denormalize(dat_hr.cpu().numpy(),varm_hr)
            
            # get mask for time step, one batch may cover muliple files
            # for ib in range(0,opt.batch_size):
            # ib = int(opt.batch_size/2) # use mask of the middle sample to represent the mask of this batch
            mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
            for ib in range(opt.batch_size):  # use mask for each sample/time
                it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
                indf = int(it/ntpd)  # id of ncfile 
                indt = it % ntpd  # the i th time step in a ncfile
                nc_f = test_set.files_hr[indf]
                for ichl in range(nchl):
                    mask[ib,ichl,:,:] = nc_load_all(nc_f,indt)[10] # mask at 1 time in a batch
                # nc_f = test_set.files_lr[indf]
                # mask_lr = nc_load_all(nc_f,indt)[10] # original data
                # mask_lr_ud = np.flipud(mask_lr) # dimensionless data flipped
                # dat_hr[:,:,mask_ud.copy()] = np.nan
    
            # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
            hr_restore1 = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
            hr_restore2 = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
            hr_restore3 = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
    
            # dat_lr[:,:,mask_lr_ud.copy()] = np.nan
            # hr_restore1 = interpolate_tensor(dat_lr, opt.up_factor, interpolation_function=griddata, method='cubic')
            # hr_restore2 = interpolate_tensor(dat_lr, opt.up_factor, interpolation_function=griddata, method='linear')
            # hr_restore3 = interpolate_tensor(dat_lr, opt.up_factor, interpolation_function=griddata, method='nearest')

            hr_restore1_norm0  = var_denormalize(hr_restore1.cpu().numpy(),varm_hr)
            hr_restore2_norm0  = var_denormalize(hr_restore2.cpu().numpy(),varm_hr)
            hr_restore3_norm0  = var_denormalize(hr_restore3.cpu().numpy(),varm_hr)

            if kmask == 1: 
                hr_norm0[mask] = np.nan            
                hr_restore1_norm0[mask] = np.nan
                hr_restore2_norm0[mask] = np.nan
                hr_restore3_norm0[mask] = np.nan
                # hr_norm0[:,:,mask] = np.nan            
                # hr_restore1_norm0[:,:,mask] = np.nan
                # hr_restore2_norm0[:,:,mask] = np.nan
                # hr_restore3_norm0[:,:,mask] = np.nan
            hr_all.append(hr_norm0)
            hr_restore1_all.append(hr_restore1_norm0)
            hr_restore2_all.append(hr_restore2_norm0)
            hr_restore3_all.append(hr_restore3_norm0)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore1,dat_hr,hr_restore1_norm0,hr_norm0) # ,mask
            metrics_re_bt['mse_re1'].append(mse)
            metrics_re_bt['mae_re1'].append(mae)
            metrics_re_bt['rmse_re1'].append(rmse)
            metrics_re_bt['ssim_re1'].append(ssim)
            metrics_re_bt['psnr_re1'].append(psnr)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore2,dat_hr,hr_restore2_norm0,hr_norm0) # ,mask
            metrics_re_bt['mse_re2'].append(mse)
            metrics_re_bt['mae_re2'].append(mae)
            metrics_re_bt['rmse_re2'].append(rmse)
            metrics_re_bt['ssim_re2'].append(ssim)
            metrics_re_bt['psnr_re2'].append(psnr)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore3,dat_hr,hr_restore3_norm0,hr_norm0) # ,mask
            metrics_re_bt['mse_re3'].append(mse)
            metrics_re_bt['mae_re3'].append(mae)
            metrics_re_bt['rmse_re3'].append(rmse)
            metrics_re_bt['ssim_re3'].append(ssim)
            metrics_re_bt['psnr_re3'].append(psnr)
        for key, value in metrics_re_bt.items():
            metrics_re[key] = sum(metrics_re_bt[key])/len(metrics_re_bt[key]) # / Nbatch_t
    
        hr_all = np.array(hr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
        # hr_all[:,:,mask] = np.nan
        hr_restore1_all = np.array(hr_restore1_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
        hr_restore2_all = np.array(hr_restore2_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
        hr_restore3_all = np.array(hr_restore3_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
        
        if not os.path.isfile(filename99): 
            hr_99per = np.nanpercentile(hr_all, 99, axis = (0,))
            hr_re1_99per = np.nanpercentile(hr_restore1_all, 99, axis = (0,))
            hr_re2_99per = np.nanpercentile(hr_restore2_all, 99, axis = (0,))
            hr_re3_99per = np.nanpercentile(hr_restore3_all, 99, axis = (0,))
            # save and Load 99per
            # filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
            np.savez(filename99,v0=hr_99per,v1=hr_re1_99per,v2=hr_re2_99per,v3=hr_re3_99per) 
        else:
            datald = np.load(filename99) # load
            hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']

        if not os.path.isfile(filename01): 
            hr_01per = np.nanpercentile(hr_all, 1, axis = (0,))
            hr_re1_01per = np.nanpercentile(hr_restore1_all, 1, axis = (0,))
            hr_re2_01per = np.nanpercentile(hr_restore2_all, 1, axis = (0,))
            hr_re3_01per = np.nanpercentile(hr_restore3_all, 1, axis = (0,))
            # save and Load 99per
            # filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
            np.savez(filename01,v0=hr_01per,v1=hr_re1_01per,v2=hr_re2_01per,v3=hr_re3_01per) 
        else:
            datald = np.load(filename01) # load
            hr_01per,hr_re1_01per,hr_re2_01per,hr_re3_01per = datald['v0'],datald['v1'],datald['v2'],datald['v3']

        if not os.path.isfile(filename_m): 
            hr_mean = np.nanmean(hr_all, axis = (0,))        
            hr_re1_mean = np.nanmean(hr_restore1_all, axis = (0,))
            hr_re2_mean = np.nanmean(hr_restore2_all, axis = (0,))
            hr_re3_mean = np.nanmean(hr_restore3_all, axis = (0,))
            # save and Load 99per
            # filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
            np.savez(filename_m,v0=hr_mean,v1=hr_re1_mean,v2=hr_re2_mean,v3=hr_re3_mean)             
        else:
            datald = np.load(filename_m) # load
            hr_mean,hr_re1_mean,hr_re2_mean,hr_re3_mean = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        
        
        for k in range(3): # nchl_o, only save ssh, u, v for all time steps. 
            ich = ivar_hr[k]-3            
            filename = out_path + 'c%d_'%(ich)+'hr_all'+'_train%4.2f'%(rtra)+'.npz'
            # if not os.path.isfile(filename): 
            #     var_all  = hr_all[:,k,:,:]
            #     np.savez(filename,v0=var_all)            
            # filename = out_path + 'c%d_'%(ich)+'hr_restore_all'+'_train%4.2f'%(rtra)+'.npz'
            # if not os.path.isfile(filename):
            #     var_all  = [hr_restore1_all[:,k,:,:],hr_restore2_all[:,k,:,:],hr_restore3_all[:,k,:,:]]
            #     np.savez(filename,v0=var_all)
        
        # save and Load dict
        file_metric_intp = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
        if not os.path.isfile(file_metric_intp):
            np.save(file_metric_intp, metrics_re_bt) 
        # metrics_re_bt = np.load(file_metric_intp,allow_pickle='TRUE').item()
        # print(metrics_re_bt['rmse_re3']) # displays 
    else:
        datald = np.load(filename99) # load
        hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        # hr_re1_99per = datald['v1']
        # hr_re2_99per = datald['v2']
        # hr_re3_99per = datald['v3']
        file_metric_intp = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
        metrics_re_bt = np.load(file_metric_intp,allow_pickle='TRUE').item()
        datald = np.load(filename01) # load
        hr_01per,hr_re1_01per,hr_re2_01per,hr_re3_01per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        datald = np.load(filename_m) # load
        hr_mean,hr_re1_mean,hr_re2_mean,hr_re3_mean = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    
    filename99m = out_path + 'hr_99per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
    if not os.path.isfile(filename99m): 
        rmse_99_re1 = np.nanmean((hr_re1_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
        rmse_99_re2 = np.nanmean((hr_re2_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
        rmse_99_re3 = np.nanmean((hr_re3_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
        mae_99_re1 = np.nanmean(abs(hr_re1_99per - hr_99per),axis=(1,2))
        mae_99_re2 = np.nanmean(abs(hr_re2_99per - hr_99per),axis=(1,2))
        mae_99_re3 = np.nanmean(abs(hr_re3_99per - hr_99per),axis=(1,2))
        np.savez(filename99m,v0=rmse_99_re1,v1=rmse_99_re2,v2=rmse_99_re3,
                 v3=mae_99_re1,v4=mae_99_re2,v5=mae_99_re3) 
    else:
        datald = np.load(filename99m) # load
        rmse_99_re1,rmse_99_re2,rmse_99_re3,mae_99_re1,mae_99_re2,mae_99_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
        
    
    filename01m = out_path + 'hr_01per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
    if not os.path.isfile(filename01m): 
        rmse_01_re1 = np.nanmean((hr_re1_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
        rmse_01_re2 = np.nanmean((hr_re2_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
        rmse_01_re3 = np.nanmean((hr_re3_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
        mae_01_re1 = np.nanmean(abs(hr_re1_01per - hr_01per),axis=(1,2))
        mae_01_re2 = np.nanmean(abs(hr_re2_01per - hr_01per),axis=(1,2))
        mae_01_re3 = np.nanmean(abs(hr_re3_01per - hr_01per),axis=(1,2))
        np.savez(filename01m,v0=rmse_01_re1,v1=rmse_01_re2,v2=rmse_01_re3,
                 v3=mae_01_re1,v4=mae_01_re2,v5=mae_01_re3) 
    else:
        datald = np.load(filename01m) # load
        rmse_01_re1,rmse_01_re2,rmse_01_re3,mae_01_re1,mae_01_re2,mae_01_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
        

    filename_mm = out_path + 'hr_mean_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
    if not os.path.isfile(filename_mm): 
        rmse_m_re1 = np.nanmean((hr_re1_mean - hr_mean) ** 2,axis=(1,2))**(0.5)
        rmse_m_re2 = np.nanmean((hr_re2_mean - hr_mean) ** 2,axis=(1,2))**(0.5)
        rmse_m_re3 = np.nanmean((hr_re3_mean - hr_mean) ** 2,axis=(1,2))**(0.5)
        mae_m_re1 = np.nanmean(abs(hr_re1_mean - hr_mean),axis=(1,2))
        mae_m_re2 = np.nanmean(abs(hr_re2_mean - hr_mean),axis=(1,2))
        mae_m_re3 = np.nanmean(abs(hr_re3_mean - hr_mean),axis=(1,2))
        np.savez(filename_mm,v0=rmse_m_re1,v1=rmse_m_re2,v2=rmse_m_re3,
                 v3=mae_m_re1,v4=mae_m_re2,v5=mae_m_re3) 
    else:
        datald = np.load(filename_mm) # load
        rmse_m_re1,rmse_m_re2,rmse_m_re3,mae_m_re1,mae_m_re2,mae_m_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
        
        
    txtname = out_path+'99th_interp.txt'
    if not os.path.isfile(txtname):
        outfile = open(txtname, 'w')
        outfile.write('# rmse_99_re1, rmse_99_re2,rmse_99_re3\n')
        np.savetxt(outfile, np.vstack((rmse_99_re1,rmse_99_re2,rmse_99_re3)), fmt='%-7.4f,')
        outfile.write('# mae_99_re1, mae_99_re2,mae_99_re3\n')
        np.savetxt(outfile, np.vstack((mae_99_re1,mae_99_re2,mae_99_re3)), fmt='%-7.4f,')
        outfile.write('# max hr_99per, hr_re1_99per, hr_re2_99per, hr_re3_99per\n')
        np.savetxt(outfile, np.vstack((np.nanmax(hr_99per,axis=(1,2)),
                                       np.nanmax(hr_re1_99per,axis=(1,2)),
                                       np.nanmax(hr_re2_99per,axis=(1,2)),
                                       np.nanmax(hr_re3_99per,axis=(1,2)))), fmt='%-7.4f,')
        outfile.write('# min hr_99per, hr_re1_99per, hr_re2_99per, hr_re3_99per\n')
        np.savetxt(outfile, np.vstack((np.nanmin(hr_99per,axis=(1,2)),
                                       np.nanmin(hr_re1_99per,axis=(1,2)),
                                       np.nanmin(hr_re2_99per,axis=(1,2)),
                                       np.nanmin(hr_re3_99per,axis=(1,2)))), fmt='%-7.4f,') 
        outfile.close()
        
    clim = [[[1.3,3.3],[1.3,3.3],[-0.2,0.2]],  # ssh
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # u
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # v
            [[12,15],[12,15],[-1.0,1.0]],  # uw
            [[12,15],[12,15],[-1.0,1.0]],  # vw
            [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],  # swh
            [[5.0,15],[5.0,15],[-2.0,2.0]],  # pwp
            ]
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)']

    for k in range(nchl_o):
        ich = ivar_hr[k]-3
        clim_chl = clim[ich]
        sample  = [hr_99per[k,:,:],hr_re1_99per[k,:,:],hr_re1_99per[k,:,:]-hr_99per[k,:,:]]
        unit = [unit_suv[ich]]*len(sample)
        title = ['hr_99','bicubic_99','interp-hr' +'(%5.3f'%mae_99_re1[ich]+',%5.3f'%rmse_99_re1[ich]+')']
        figname = out_path+"99th_c%d_int1.png" % (ich)
        plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title)

        sample  = [hr_99per[k,:,:],hr_re2_99per[k,:,:],hr_re2_99per[k,:,:]-hr_99per[k,:,:]]
        title = ['hr_99','bilinear_99','interp-hr'+'(%5.3f'%mae_99_re2[ich]+',%5.3f'%rmse_99_re2[ich]+')']
        figname = out_path+"99th_c%d_int2.png" % (ich)
        plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title)
        
        sample  = [hr_99per[k,:,:],hr_re3_99per[k,:,:],hr_re3_99per[k,:,:]-hr_99per[k,:,:]]
        title = ['hr_99','nearest_99','interp-hr'+'(%5.3f'%mae_99_re3[ich]+',%5.3f'%rmse_99_re3[ich]+')']
        figname = out_path+"99th_c%d_int3.png" % (ich)
        plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title)

