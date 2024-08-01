#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import numpy as np
import os
import time

import torch
from PIL import Image
import torchvision.transforms as transforms

from models import GeneratorResNet
from datasets import hr_transform,lr_transform,find_maxmin_global
from funs_prepost import (nc_var_normalize,nc_load_all,var_denormalize,plot_sites_cmpn)

from datetime import datetime, timedelta # , date
from funs_sites import index_stations

def lst_flatten(xss):
    return [x for xs in xss for x in xs]

def select_sta(files_hr,ivar_hr,files_lr,ivar_lr,ntpd):
    
    nc_f = files_hr[0]
    lon = nc_load_all(nc_f)[1]
    lat = nc_load_all(nc_f)[2]
    
    # estimate max min value for the selected period
    var_lst = ['s','u','v']
    varm_hr_test,ind_varm,file_varm = find_maxmin_global(files_hr, ivar_hr)
    varm_lr_test,_,_ = find_maxmin_global(files_lr, ivar_lr) # ssh(Nt,Nlat,Nlon)
    sta_max = lst_flatten([['max'+var_lst[ivar_hr[i]-3], 'min'+var_lst[ivar_hr[i]-3]] for i in range(nchl_o)])
    temp = np.unravel_index(ind_varm.flatten(), (ntpd,len(lat),len(lon)))
    index = np.array([temp[1],temp[2]]).transpose()
    
    # # select several tidal gauge locations for comparison 
    sta_TG0 = ['CuxhavenTG','BremerhavenTG','WilhelmshavenTG','AlteWeserTG']
    sta_TG = [sta_TG0[i] + str(j) for i in range(len(sta_TG0)) for j in range(4)] # add neighbor points
    ll_TG0 = np.array([[8.717500,53.867800],[8.568000,53.545000],[8.145000,53.514400],[8.127500,53.863300]])
    ll_shift = np.array([[0,0],[0,0],[0,0],[0,0]]) # shift the station to the water region, lon,lat
    ll_stas = ll_TG0+ll_shift
    ind_TG = index_stations(ll_stas[:,0],ll_stas[:,1],lon,lat)
    # index = ind_sta
    
    # add points in the domain for testing
    # nx_smp = 8
    # ny_smp = 8
    # ix = np.linspace(0, len(lat)-1, nx_smp)
    # iy = np.linspace(0, len(lon)-1, ny_smp)
    # ix = np.arange(0, len(lat)-1, 16) #  shared poins by lr & hr
    # iy = np.arange(0, len(lon)-1, 16)
    ix = np.arange(8, len(lat)-1, 16) #  non-shared poins by lr & hr, initial index links to scale
    iy = np.arange(8, len(lon)-1, 16)
    xv, yv = np.meshgrid(ix, iy)
    ind_add = np.vstack((np.int_(xv.flatten()), np.int_(yv.flatten()))).T
    sta_add = ['p'+ str(i).zfill(2) for i in range(len(ind_add))]
    
    # add gauging stations 
    # index = np.vstack((index, ind_add, ind_TG))
    # sta_user = sta_max + sta_add + sta_TG
    # index = np.vstack((index, ind_add)) # no gauging station
    # sta_user = sta_max + sta_add 
    index = ind_TG # only gauging station
    sta_user = sta_TG 
    ll_sta = np.array([lat[index[:,0]],lon[index[:,1]]]).transpose() # should corresponds to (H,W), H[0]-lowest lat
    return index,sta_user,ll_sta,varm_hr_test,ind_varm,varm_lr_test

kmask = 1

import importlib
mod_name= 'par01'
mod_para=importlib.import_module(mod_name)

if __name__ == '__main__':
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    ivar_lr = mod_para.ivar_lr
    ivar_hr = mod_para.ivar_hr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(ivar_lr)
    nchl_o = len(ivar_hr)
    test_dir_lr = opt.dir_lr
    test_dir_hr = opt.dir_hr
    
    nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    rep = [4]
    epoc_num =[97] #
    key_ep_sort = 0 # to use epoc here or load sorted epoc no. 
    nepoc = 3 # no. of sorted epochs for analysis

    keyplot = 1
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    
    # select a range of data for testing 
    tlim = [datetime(2017,10,27),datetime(2017,11,1)]
    # tlim = [datetime(2017,1,10),datetime(2017,1,13)]
    # tlim = [datetime(2018,1,2),datetime(2018,1,6)]
    tlim = [datetime(2017,11,18),datetime(2017,11,21)]
    # tlim = [datetime(2017,12,7),datetime(2017,12,10)]
    # tlim = [datetime(2017,1,16),datetime(2017,10,8)]
    tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d')
    
    nday = (tlim[1] - tlim[0]).days
    iday0 = (tlim[0] - datetime(2017,1,2)).days+1 # out2d_interp_001.nc corresponds to 2017.1.2
    iday1 = (tlim[1] - datetime(2017,1,2)).days+1
    dif = int((tlim[1]-tlim[0]).total_seconds()/3600) ## time difference in hours
    tuser0 = [(tlim[0] + timedelta(hours=x)) for x in range(0,dif)]
    tshift = 2 # in hour
    tuser = [(tlim[0] + timedelta(hours=x)) for x in range(tshift,dif+tshift)] # time shift for numerical model
    id_test = np.arange(iday0,iday1)

    # id_test = np.arange(iday1,400)
    files_lr = [test_dir_lr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]
    files_hr = [test_dir_hr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]
    nfile = len(files_lr)
    
    # get mask of data, this is for the initial file
    # nc_f = files_hr[0]
    # mask = nc_load_all(nc_f,0)[10] # original data

    ntpd = int(24)
    index,sta_user,ll_sta,varm_hr_test,ind_varm,varm_lr_test = select_sta(files_hr,ivar_hr,files_lr,ivar_lr,ntpd)
    
    out_path0 = 'results_pnt/'+'SRF_'+str(opt.up_factor)+suf
    os.makedirs(out_path0, exist_ok=True)    

    
    for irep in rep:
        if key_ep_sort:
            # choose sorted epoc number that gives the smallest rmse
            ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort_rmse.csv' #  rank based on 99 percentile, only saved for last var
            # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort'+tstr+'.csv' # rank based on rt_use highest ssh/
            ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
            epoc_num = ep_sort.flatten()[0:nepoc*nchl_o] 
            epoc_num = list(set(epoc_num.tolist())) # remove duplicates,order not reseved
            
        out_path = 'results_pnt/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)
        os.makedirs(out_path, exist_ok=True)
        
        txtname = out_path0+"/para_cmp"+'_rp%d_ep%d_%d'%(irep,epoc_num[0],epoc_num[-1])+tstr+'.txt'
        outfile = open(txtname, 'w')
        outfile.write('# varm_hr_test, varm_lr_test\n')
        np.savetxt(outfile, np.hstack((varm_hr_test,varm_lr_test)), fmt='%-7.4f,')
        outfile.write('# irep: {0}\n'.format(irep))
        
        for epoc in epoc_num:

            outfile.write('# epo {0}\n'.format(epoc))
            # suf = '_res' + str(opt.residual_blocks) + '_max_var1para_r8' # + '_eps_'+str(0)

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
            
            # hr_scale = transforms.Resize((opt.hr_height, opt.hr_width), Image.Resampling.BICUBIC,antialias=None)
            
            nsta = len(index)
            sr_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_res1_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_res2_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_res3_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            
            sr_varm = np.zeros(shape=(nchl_o,2,len(tuser0)))
            hr_varm = np.zeros(shape=(nchl_o,2,len(tuser0)))
            dif_varm = np.zeros(shape=(nchl_o,2,len(tuser0)))
            
            for indf in range(0,nfile):
                nc_f_lr = files_lr[indf % len(files_lr)]
                nc_f_hr = files_hr[indf % len(files_hr)]
                nc_id = int(nc_f_lr.split('/')[-1][-6:-3])
                for indt in range(0,ntpd): # single image
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
                    
                    hr_norm0[:,:,mask] = np.nan
                    mse = np.nanmean((sr_norm0 - hr_norm0) ** 2,axis=(0,2,3)) # mean for channel
            
                    it = indt+indf*ntpd
                    sr_varm[:,0,it] = sr_norm0.max(axis=(0,2,3)) # max for channel
                    sr_varm[:,1,it] = sr_norm0.min(axis=(0,2,3)) # min for channel
                    hr_varm[:,0,it] = np.nanmax(hr_norm0,axis=(0,2,3)) # max for channel
                    hr_varm[:,1,it] = np.nanmin(hr_norm0,axis=(0,2,3)) # min for channel
                    dif = sr_norm0 - hr_norm0
                    dif_varm[:,0,it] = np.nanmax(dif,axis=(0,2,3)) # max for channel
                    dif_varm[:,1,it] = np.nanmin(dif,axis=(0,2,3)) # min for channel
                    
                    for ip in range(nsta):
                        sr_sta[ip,:,it]=sr_norm0[:,:,index[ip,0],index[ip,1]]
                        hr_sta[ip,:,it]=hr_norm0[:,:,index[ip,0],index[ip,1]]
                        hr_res1_sta[ip,:,it]=hr_restore1_norm0[:,:,index[ip,0],index[ip,1]]
                        hr_res2_sta[ip,:,it]=hr_restore2_norm0[:,:,index[ip,0],index[ip,1]]
                        hr_res3_sta[ip,:,it]=hr_restore3_norm0[:,:,index[ip,0],index[ip,1]]
                        
            sr_varmt = np.array([sr_varm[:,0,:].max(axis=1), sr_varm[:,1,:].min(axis=1)]).transpose()
            hr_varmt = np.array([hr_varm[:,0,:].max(axis=1), hr_varm[:,1,:].min(axis=1)]).transpose()
            dif_varmt = np.array([dif_varm[:,0,:].max(axis=1), dif_varm[:,1,:].min(axis=1)]).transpose()
            
            outfile.write('# stations sr_varmt, hr_varmt, dif_varmt\n')
            np.savetxt(outfile, np.hstack((sr_varmt,hr_varmt,dif_varmt)), fmt='%-7.4f,')
            
            index_max = np.argmax(hr_sta,axis=2)
            index_min = np.argmin(hr_sta,axis=2)
            
            if keyplot == 1:
                # plot comparison for locations
                axlab = [['Time','ssh (m)'],['Time','u (m/s)'],['Time','v (m/s)']]
                leg = ['hr','sr','bicubic','bilinear','nearest']
                # line_sty=['k.','b','r-','m-','g-','c']
                line_sty=['ko','b','r-','m-','g-','c'] # 'kv',
                for ip in range(nsta):
                    for i in range(len(ivar_hr)):
                        var_sta = hr_sta[ip,i,:]
                        var = sr_sta[ip,i,:]
                        var_res1 = hr_res1_sta[ip,i,:]
                        var_res2 = hr_res2_sta[ip,i,:]
                        var_res3 = hr_res3_sta[ip,i,:]
                        time_lst = [tuser0,tuser0,tuser0,tuser0,tuser0]
                        date_lst = [var_sta,var,var_res1,var_res2,var_res3]
                        ich = ivar_hr[i]-3
                        # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoc) +tstr+ sta_user[ip]+'.png'
                        figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoc) +tstr+'_ll%4.3f_%4.3f_'%(ll_sta[ip,1],ll_sta[ip,0])+ sta_user[ip]+'.png'
                        plot_sites_cmpn(time_lst,date_lst,tlim,figname,axlab[ich],leg=leg,leg_col=2,line_sty=line_sty)
                    
        outfile.close()
