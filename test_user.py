#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import numpy as np
import os
import time
import glob

import torch
from PIL import Image
# import torchvision.transforms as transforms

from models import GeneratorResNet
from datasets import hr_transform,lr_transform
from funs_prepost import (nc_var_normalize,var_denormalize,find_max_global,
                      nc_load_all,plt_pcolor_list,plot_sites_cmpn)

from math import log10
import pandas as pd
# import pytorch_ssim
from pytorch_msssim import ssim as ssim_torch
# from skimage.metrics import peak_signal_noise_ratio as psnr_skimg
# from  skimage.metrics import structural_similarity as ssim_skimg
from datetime import datetime, timedelta # , date


def cal_metrics(pr,hr,pr_norm0,hr_norm0,mask): # pr,hr are tensors [N,C,H,W], norm0 are arrays,
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

kmask = 1

import sys
import importlib
mod_name= 'par01_s64_'  #'par01' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
opt = mod_para.opt
suf = mod_para.suf+mod_name
rtra = mod_para.rtra
ivar_lr = mod_para.ivar_lr
ivar_hr = mod_para.ivar_hr
varm_hr = mod_para.varm_hr
varm_lr = mod_para.varm_lr
nchl_i = len(ivar_lr)
nchl_o = len(ivar_hr)
dir_lr = opt.dir_lr
dir_hr = opt.dir_hr

nrep = mod_para.nrep
nrep = 1
rep = list(range(0,nrep))
rep =[2]

epoc_num =[84] # epoch used for testing 
# epoc_num = np.arange(40,opt.N_epochs+1)
key_ep_sort = 0 # to use epoc here or load sorted epoc no. 
nepoc = 1 # no. of sorted epochs for analysis
keyplot = 1

rt_use = 0.01 # use top largest values of testing data for testing 
tstr = '_rk%4.2f'%rt_use

opath_st = 'statistics'+ suf  +'_mk'+str(kmask) +'/'
opath_st_hr = 'statistics_hr'+'/' 
if not os.path.exists(opath_st_hr):
    os.makedirs(opath_st_hr)
    
dateref = datetime(2017,1,2) # out2d_interp_001.nc corresponds to 2017.1.2

# # select a range of data for testing 
# tlim = [datetime(2017,10,26),datetime(2017,11,1)]
# tlim = [datetime(2017,1,10),datetime(2017,1,13)]
# nday = (tlim[1] - tlim[0]).days
# iday0 = (tlim[0] - dateref).days+1
# iday1 = (tlim[1] - dateref).days+1
# dif = int((tlim[1]-tlim[0]).total_seconds()/3600) ## time difference in hours
# tuser0 = [(tlim[0] + timedelta(hours=x)) for x in range(0,dif)]
# # tshift = 2 # in hour
# # tuser = [(tlim[0] + timedelta(hours=x)) for x in range(tshift,dif+tshift)] # time shift for numerical model
# id_test = np.arange(iday0,iday1)
# tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d')


files_lr = sorted(glob.glob(dir_lr + "/*.nc"))
files_hr = sorted(glob.glob(dir_hr + "/*.nc"))
nfile = len(files_hr)
ind_train = np.arange(15,int(nfile*rtra)+15)    # exclude max
ind_valid= np.delete(np.arange(0,nfile),ind_train)
files_lr = [files_lr[i] for i in ind_valid]
files_hr = [files_hr[i] for i in ind_valid]

ichlo = ivar_hr[0]-3 # only work for one variable
filename = opath_st_hr+'var%d'%ichlo+'_sorted_test'+dir_hr.split("_",1)[1]+'_rt%4.2f'%(rtra)+'.npz'
if not os.path.isfile(filename):
    var_sort,ind_sort = find_max_global(files_hr, ivar=[ivar_hr[0]]) # find maximum index 
    np.savez(filename, v1=var_sort,v2=ind_sort) 
datald = np.load(filename) # load
nhr = datald['v1'].size
var_sort = datald['v1'][0][0:int(nhr*rt_use)]
ind_sort = datald['v2'][0][0:int(nhr*rt_use)]  # index of maximum ssh 
id_test = ind_sort

# id_test = np.arange(380,381)
# id_test = np.arange(300,306)
# id_test = np.arange(302,303)
# id_test = np.arange(9,12)
# id_test = np.arange(202,203)

int_mod = '' # 'bicubic', 'nearest', 'bilinear'

# get logitude and latitude of data 
nc_f = files_hr[0]
lon = nc_load_all(nc_f,0)[1]
lat = nc_load_all(nc_f,0)[2]
mask0 = nc_load_all(nc_f,0)[10] # original data
    
# nearest,bicubit, sr, GT, diff,diff,diff
clim = [[[1.3,3.3],[1.3,3.3],[1.3,3.3],[1.3,3.3],[-0.2,0.2],[-0.2,0.2],[-0.2,0.2]],  # ssh
        [[0.2,1.8],[0.2,1.8],[0.2,1.8],[0.2,1.8],[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]],  # u
        [[0.2,1.8],[0.2,1.8],[0.2,1.8],[0.2,1.8],[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]],  # v
        [[12,15],[12,15],[12,15],[12,15],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],  # uw
        [[12,15],[12,15],[12,15],[12,15],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],  # vw
        [[2.0,5.0],[2.0,5.0],[2.0,5.0],[2.0,5.0],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]],  # swh
        [[5.0,15.],[5.0,15.],[5.0,15.],[5.0,15.],[-2.0,2.0],[-2.0,2.0],[-2.0,2.0]],  # pwp
        ]
unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)']

#  make a list for figure captions
alpha = list(map(chr, range(ord('a'), ord('z')+1)))
capt_all = ['('+alpha[i]+')' for i in range(len(alpha))]

for irep in rep:
# irep = 0
    print(f'Repeat {irep}')
    print('--------------------------------')
    
    # suf = '_res' + str(opt.residual_blocks) + '_max_var1_nb04'
    out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)
    os.makedirs(out_path, exist_ok=True)
    
    if key_ep_sort:
        # choose sorted epoc number that gives the smallest rmse_99
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort.csv' #  rank based on rmse99 
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort'+tstr+'.csv' # rank based on rt_use highest ssh/
        ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort_rmse.csv' #  rank based on rmse
        ep_sort1 = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
        epoc_num = np.concatenate([ep_sort.flatten()[0:nepoc*nchl_o],ep_sort1.flatten()[0:nepoc*nchl_o]])
        epoc_num = list(set(epoc_num.tolist())) # remove duplicates,order not reseved

    metrics = {'ep':[],'mae': [], 'mse': [], 'rmse': [], 'psnr': [], 'ssim': [],
               'mae_re1': [], 'mse_re1': [], 'rmse_re1': [], 'psnr_re1': [], 'ssim_re1': [],
               'mae_re2': [], 'mse_re2': [], 'rmse_re2': [], 'psnr_re2': [], 'ssim_re2': [],
               'mae_re3': [], 'mse_re3': [], 'rmse_re3': [], 'psnr_re3': [], 'ssim_re3': [],
               }
    metrics_chl = {}
    
    for epoc in epoc_num:
    
        opath_nn = 'nn_models_' + str(opt.up_factor) + suf +'/' # 
        model_name = 'netG_epoch_%d_re%d.pth' % (epoc,irep)
        # TEST_MODE = True if opt.test_mode == 'GPU' else False
        
        model = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor).eval()
        if torch.cuda.is_available():
            print('Use GPU')
            model.cuda()
            checkpointG = torch.load(opath_nn+ model_name)
        else:
            print('Use CPU')
            checkpointG = torch.load(opath_nn+ model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpointG['model_state_dict'])
        
        # files_lr = [test_dir_lr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]
        # files_hr = [test_dir_hr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]
        # nfile = len(files_lr)
        ntpd = 24 
        
        metrics_it = {'mae': [], 'mse': [], 'rmse': [], 'psnr': [], 'ssim': [],
                   'mae_re1': [], 'mse_re1': [], 'rmse_re1': [], 'psnr_re1': [], 'ssim_re1': [],
                   'mae_re2': [], 'mse_re2': [], 'rmse_re2': [], 'psnr_re2': [], 'ssim_re2': [],
                   'mae_re3': [], 'mse_re3': [], 'rmse_re3': [], 'psnr_re3': [], 'ssim_re3': [],
                   }
        id_smp = {'day': [], 'hour':[]}
        metrics_it_chl = {}
        # hr_scale = transforms.Resize((opt.hr_height, opt.hr_width), Image.Resampling.BICUBIC,antialias=None)
        
        nchl = nchl_o
        
        rk = 0
        for it in ind_sort: # hours from the first file (reference time)
            indf = int(it/ntpd)  # id of ncfile that contain field with id index
            indt = it % ntpd  # the i th time step in a ncfile
            nc_f_lr = files_lr[indf % len(files_lr)]
            nc_f_hr = files_hr[indf % len(files_hr)]
            nc_id = int(nc_f_lr.split('/')[-1][-6:-3])
            rk = rk +1
            
            # get mask of data at specified time 
            mask = nc_load_all(nc_f_hr,indt)[10] # original data
            
            data = nc_var_normalize(nc_f_hr,indt,ivar_hr,varm_hr) # flipud height for image
            # img = Image.fromarray((data * 255).astype(np.uint8)) # RGB
            # hr = hr_transform(opt.hr_height, opt.hr_width)(img)
            x = []
            for i in range(data.shape[2]): # data(H,W,C)
                img = Image.fromarray(data[:,:,i]) # single channel to img
                x.append(hr_transform(opt.hr_height, opt.hr_width)(img))
            hr = torch.cat(x)
            
            data = nc_var_normalize(nc_f_lr,indt,ivar_lr,varm_lr)
            # img = Image.fromarray((data * 255).astype(np.uint8)) # RGB
            # lr = lr_transform(opt.hr_height, opt.hr_width, opt.up_factor)(img)
            x = []
            for i in range(data.shape[2]): # data(H,W,C)
                img = Image.fromarray(data[:,:,i]) # single channel to img
                x.append(lr_transform(opt.hr_height, opt.hr_width, opt.up_factor)(img))
            lr = torch.cat(x)
            
            if torch.cuda.is_available():
                lr = lr.cuda()
                
            id_smp['day'].append(nc_id)
            id_smp['hour'].append(indt)    
                    
            # hr_restore = hr_scale(lr)
            
            lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
            hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
            # hr_restore = hr_restore.reshape(1,hr_restore.shape[0],hr_restore.shape[1],hr_restore.shape[2]) # 3d to 4d
    
            # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
            hr_restore1 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
            hr_restore2 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
            hr_restore3 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
    
            start = time.time()
            sr = model(lr)
            end = time.time()
            elapsed = (end - start)
            print('cost ' + str(elapsed) + 's')
    
            sr_norm0 = var_denormalize(sr.detach().numpy(),varm_hr) # (N,C,H,W), flipud height back
            hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
            hr_restore1_norm0  = var_denormalize(hr_restore1.detach().numpy(),varm_hr)
            hr_restore2_norm0  = var_denormalize(hr_restore2.detach().numpy(),varm_hr)
            hr_restore3_norm0  = var_denormalize(hr_restore3.detach().numpy(),varm_hr)
            
            if kmask==1:
                hr_norm0[:,:,mask] = np.nan
                sr_norm0[:,:,mask] = np.nan
                hr_restore1_norm0[:,:,mask] = np.nan
                hr_restore2_norm0[:,:,mask] = np.nan
                hr_restore3_norm0[:,:,mask] = np.nan
            
            rmse, mae, mse, ssim, psnr = cal_metrics(sr,hr,sr_norm0,hr_norm0,mask)
            metrics_it['ssim'].append(ssim)
            metrics_it['psnr'].append(psnr) 
            metrics_it['mse'].append(mse)
            metrics_it['mae'].append(mae)
            metrics_it['rmse'].append(rmse)
            
            # for restored hr
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore1,hr,hr_restore1_norm0,hr_norm0,mask)
            metrics_it['ssim_re1'].append(ssim)
            metrics_it['psnr_re1'].append(psnr) 
            metrics_it['mse_re1'].append(mse)
            metrics_it['mae_re1'].append(mae)
            metrics_it['rmse_re1'].append(rmse)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore2,hr,hr_restore2_norm0,hr_norm0,mask)
            metrics_it['ssim_re2'].append(ssim)
            metrics_it['psnr_re2'].append(psnr) 
            metrics_it['mse_re2'].append(mse)
            metrics_it['mae_re2'].append(mae)
            metrics_it['rmse_re2'].append(rmse)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore3,hr,hr_restore3_norm0,hr_norm0,mask)
            metrics_it['ssim_re3'].append(ssim)
            metrics_it['psnr_re3'].append(psnr) 
            metrics_it['mse_re3'].append(mse)
            metrics_it['mae_re3'].append(mae)
            metrics_it['rmse_re3'].append(rmse)
            
            if keyplot == 1:
            # dat_grid = torch.cat((hr_restore, hr.cpu(), sr.cpu()), -1)
                # nsmp = indf*ntpd+indt
                # tsri = tuser0[nsmp].strftime('%Y%m%d_%H')
                tp = dateref + timedelta(hours=int(it))
                tsri = tp.strftime('%Y%m%d_%H')
                dat_grid = torch.cat((hr.cpu(), sr.cpu(),hr_restore1,hr_restore2,hr_restore3), -1)
                ncol = 1
                sample = dat_grid
                sample = sample.detach().numpy()
                for ich in range(nchl):
                    # figname = out_path+"/c%d_epoch%d_rk%03d" % (ivar_hr[ich]-3,epoc,rk) + tsri+'.png'
                    # plt_sub(sample,ncol,figname,ich)                     
                    ichl = ivar_hr[ich]-3
                    clim_chl = clim[ichl]                
                    sample  = [hr_restore3_norm0[0,ich,:,:],
                               hr_restore2_norm0[0,ich,:,:],
                               sr_norm0[0,ich,:,:],
                               hr_norm0[0,ich,:,:],
                               hr_restore3_norm0[0,ich,:,:]-hr_norm0[0,ich,:,:],
                               hr_restore2_norm0[0,ich,:,:]-hr_norm0[0,ich,:,:],
                               sr_norm0[0,ich,:,:]-hr_norm0[0,ich,:,:],
                               ]
                    unit = [unit_suv[ichl]]*len(sample)
                    title = ['nearest','bilinear','sr','hr',
                             'nearest-hr', # +'(%5.3f'%metrics_it['mae_re3'][rk-1][ich]+',%5.3f'%metrics_it['rmse_re3'][rk-1][ich]+')',
                             'bilinear-hr', # +'(%5.3f'%metrics_it['mae_re2'][rk-1][ich]+',%5.3f'%metrics_it['rmse_re2'][rk-1][ich]+')',
                             'sr-hr', # +'(%5.3f'%metrics_it['mae'][rk-1][ich]+',%5.3f'%metrics_it['rmse'][rk-1][ich]+')',
                             ]
                    txt = ['','','','',
                           'MAE=%5.3f'%metrics_it['mae_re3'][rk-1][ich]+'\nRMSE=%5.3f'%metrics_it['rmse_re3'][rk-1][ich],
                           'MAE=%5.3f'%metrics_it['mae_re2'][rk-1][ich]+'\nRMSE=%5.3f'%metrics_it['rmse_re2'][rk-1][ich],
                           'MAE=%5.3f'%metrics_it['mae'][rk-1][ich]+'\nRMSE=%5.3f'%metrics_it['rmse'][rk-1][ich],
                           ]
                    loc_txt = [0.52,0.40] # location of text
                    figname = out_path+"/c%d_rk%02d_ep%d_" % (ivar_hr[ich]-3,rk,epoc) + tsri+'dmdf_ax0.png'
                    plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,
                                    unit=unit,title=title,nrow=2,axoff=1,capt=capt_all,txt=txt,loc_txt=loc_txt)
                    # figname = out_path+"/c%d_rk%02d_ep%d_" % (ivar_hr[ich]-3,rk,epoc) + tsri+'dmdf.png'
                    # plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title,nrow=2)

        # axlab = [['Time','rmse(ssh) (m)'],['Time','rmse(u) (m/s)'],['Time','rmse(v) (m/s)']]
        axlab_rmse = [['sample','rmse(ssh) (m)'],['sample','rmse(u) (m/s)'],['sample','rmse(v) (m/s)']]
        axlab_mae = [['sample','mae(ssh) (m)'],['sample','mae(u) (m/s)'],['sample','mae(v) (m/s)']]
        
        leg = ['sr','bicubic','bilinear','nearest']
        line_sty=['k','b','r','m','g--','c']
        for i in range(nchl):
            ichl = ivar_hr[i]-3
            for key, value in metrics_it.items():
                metrics_it_chl[key] = [value[j][i] for j in range(0,len(value))]
            data_frame = pd.DataFrame.from_dict(id_smp|metrics_it_chl, orient='index').transpose()
            ofname = "c%d_epoch%d" % (ichl,epoc) +tstr+'.csv'
            data_frame.to_csv(out_path + os.sep + ofname, index_label='sample')
            
            time_lst = [list(range(1,len(ind_sort)+1))]*4 # tuser0

            var = np.array(metrics_it['rmse'])[:,i]
            var_res1 = np.array(metrics_it['rmse_re1'])[:,i]
            var_res2 = np.array(metrics_it['rmse_re2'])[:,i]
            var_res3 = np.array(metrics_it['rmse_re3'])[:,i]
            data_lst = [var,var_res1,var_res2,var_res3]
            figname = out_path+"/c%d_epoch%d" % (ichl,epoc) +tstr+'_rmse.png'
            plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab_rmse[i],leg=leg,leg_col=2)

            var = np.array(metrics_it['mae'])[:,i]
            var_res1 = np.array(metrics_it['mae_re1'])[:,i]
            var_res2 = np.array(metrics_it['mae_re2'])[:,i]
            var_res3 = np.array(metrics_it['mae_re3'])[:,i]
            data_lst = [var,var_res1,var_res2,var_res3]
            figname = out_path+"/c%d_epoch%d" % (ichl,epoc) +tstr+'_mae.png'
            plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab_mae[i],leg=leg,leg_col=2)
        
            # var = np.array(metrics_it['psnr'])[:,i]
            # var_res1 = np.array(metrics_it['psnr_re1'])[:,i]
            # var_res2 = np.array(metrics_it['psnr_re2'])[:,i]
            # var_res3 = np.array(metrics_it['psnr_re3'])[:,i]
            # data_lst = [var,var_res1,var_res2,var_res3]
            # figname = out_path+"/c%d_epoch%d" % (ichl,epoc) +tstr+'_psnr.png'
            # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=['sample','psnr'],leg=leg,leg_col=2)
            
            # var = np.array(metrics_it['ssim'])[:,i]
            # var_res1 = np.array(metrics_it['ssim_re1'])[:,i]
            # var_res2 = np.array(metrics_it['ssim_re2'])[:,i]
            # var_res3 = np.array(metrics_it['ssim_re3'])[:,i]
            # data_lst = [var,var_res1,var_res2,var_res3]
            # figname = out_path+"/c%d_epoch%d" % (ichl,epoc) +tstr+'_ssim.png'
            # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=['sample','ssim'],leg=leg,leg_col=2)    
    
    
        # save metrics time average of selected hours 
        for key, value in metrics_it.items():
            metrics[key].append(sum(metrics_it[key])/len(metrics_it[key])) #
        metrics['ep'].append([epoc]*nchl)
    
    # output metrics for all epochs to csv
    for i in range(nchl):
        for key, value in metrics.items():
            metrics_chl[key] = [value[j][i] for j in range(0,len(value))]
        data_frame = pd.DataFrame.from_dict(metrics_chl, orient='index').transpose()
        ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics'+tstr+'.csv'
        data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
    
    if key_ep_sort==0:  # if input non-sorted epoch 
        # output sorted metrics for all epochs to csv
        ind_sort = [[]]*nchl
        metrics_chl_sort = {}
        # list rmse from small to large
        for i in range(nchl):
            var = [metrics['rmse'][j][i] for j in range(len(metrics['rmse']))]
            ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
            for key, value in metrics.items():
                metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
            data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
            ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics_sort'+tstr+'.csv'
            data_frame.to_csv(opath_st + ofname, index_label='Epoch')
            
        # output epoc oder for channels
        ep_sort = [[]]*nchl
        for i in range(nchl):
            ep_sort[i] = [metrics['ep'][ind_sort[i][j]][i] for j in range(len(metrics['ep']))]
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_epoc_sort'+tstr+'.csv'
        np.savetxt(opath_st + ofname, ep_sort, fmt='%d',delimiter=",")
        # ep_sort = np.loadtxt(opath_st + ofname, dtype=int)
    