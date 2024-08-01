#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:58:54 2024
read metrics from repeated runs 
@author: g260218
"""

import os
import numpy as np

import pandas as pd
import itertools
# read metrics

import sys
import importlib
mod_name= 'par01_lr5e4'          #'par04' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
kmask = 1

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
    # rep = list(range(0,9))
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    # epoc_num = [50,100]
    epoc_num = np.arange(40,opt.N_epochs+1)

    nchl = nchl_o
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    if not os.path.exists(opath_st):
        os.makedirs(opath_st)
        
    metrics_re_bt_chl = {}
    out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+'_mask'+str(kmask)+'/'
    # load metrics along batch for direct interpolation 
    filename = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
    metrics_re_bt = np.load(filename,allow_pickle='TRUE').item()
    filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
    datald = np.load(filename) # load
    hr_99per = datald['v0']
    
    
    metrics_rp = {'rp':[],'ep':[],'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], 'mae_99': [],'rmse_99': [],} # eopch mean
    metrics_chl_rp = {}

    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'
        # np.save(opath_st + ofname, metrics) 
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics_bt.npy'
        # # np.save(opath_st + ofname, metrics_bt) 
        # metrics_bt = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        
        for i in range(len(metrics['rmse'])):
            metrics['rmse'][i] = metrics['mse'][i]**0.5
        for key, value in metrics.items():
            metrics_rp[key].extend(metrics[key]) # / Nbatch_t) # 
        metrics_rp['rp'].extend([[irep]*nchl for i in range(len(metrics['rmse']))])
        
        # output epochs based on rmse instead of rmse99 
        ind_sort = [[]]*nchl
        metrics_chl_sort = {}
        # list rmse from small to large
        for i in range(nchl):
            var = [metrics['rmse'][j][i] for j in range(len(metrics['rmse']))]
            ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
            for key, value in metrics.items():
                metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
            data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
            ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics_sort_rmse.csv'
            data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
        ep_sort = [[]]*nchl
        for i in range(nchl):
            ep_sort[i] = [metrics['ep'][ind_sort[i][j]][i] for j in range(len(metrics['ep']))]
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_epoc_sort_rmse.csv'
        np.savetxt(opath_st + ofname, np.array(ep_sort).transpose(), fmt='%d',delimiter=",")
        
    # output sorted metrics for all epochs to csv
    ind_sort = [[]]*nchl
    metrics_chl_sort = {}
    # list rmse99 from small to large
    for i in range(nchl):
        var = [metrics_rp['rmse_99'][j][i] for j in range(len(metrics_rp['rmse_99']))]
        ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
        for key, value in metrics_rp.items():
            metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
        data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
        ofname = "srf_%d_reAll_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics_sort.csv'
        data_frame.to_csv(opath_st + ofname, index_label='Epoch')

    # output epoc oder for channels rmse_99
    ep_sort = [[]]*nchl
    rp_sort = [[]]*nchl
    for i in range(nchl):
        ich = ivar_hr[i]-3
        ep_sort[i] = [metrics_rp['ep'][ind_sort[i][j]][i] for j in range(len(metrics_rp['ep']))]
        rp_sort[i] = [metrics_rp['rp'][ind_sort[i][j]][i] for j in range(len(metrics_rp['rp']))]
        ofname = "srf_%d_reAll_c%d_ep%d_%d_mask" % (opt.up_factor,ich,epoc_num[0],epoc_num[-1]) + '_epoc_sort.csv'
        combined_ind= np.column_stack((np.array(rp_sort[i]), np.array(ep_sort[i])))
        np.savetxt(opath_st + ofname, combined_ind, fmt='%d',delimiter=",")
        # ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")    
        
    # list rmse from small to large
    for i in range(nchl):
        var = [metrics_rp['rmse'][j][i] for j in range(len(metrics_rp['rmse']))]
        ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
        for key, value in metrics_rp.items():
            metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
        data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
        ofname = "srf_%d_reAll_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics_sort_rmse.csv'
        data_frame.to_csv(opath_st + ofname, index_label='Epoch')
        
    # output epoc oder for channels
    ep_sort = [[]]*nchl
    rp_sort = [[]]*nchl
    for i in range(nchl):
        ich = ivar_hr[i]-3
        ep_sort[i] = [metrics_rp['ep'][ind_sort[i][j]][i] for j in range(len(metrics_rp['ep']))]
        rp_sort[i] = [metrics_rp['rp'][ind_sort[i][j]][i] for j in range(len(metrics_rp['rp']))]
        ofname = "srf_%d_reAll_c%d_ep%d_%d_mask" % (opt.up_factor,ich,epoc_num[0],epoc_num[-1]) + '_epoc_sort_rmse.csv'
        combined_ind= np.column_stack((np.array(rp_sort[i]), np.array(ep_sort[i])))
        np.savetxt(opath_st + ofname, combined_ind, fmt='%d',delimiter=",")
        # ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")    