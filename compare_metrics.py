#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:24:51 2024
compare metrics 
@author: g260218
"""

import os
import numpy as np

import pandas as pd
import sys
import importlib

# load max rmse from sorted data from repeated runs
def read_rmse_max(mod_name,epoc_num,kmask):
    mod_para=importlib.import_module(mod_name)
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    ivar_hr = mod_para.ivar_hr
    nchl = len(ivar_hr)
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    metrics_rmse, metrics_rmse8 = [],[]
    for i in range(nchl):
        ofname = "srf_%d_reAll_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics_sort_rmse.csv'
        metrics = np.loadtxt(opath_st + ofname, delimiter=",",skiprows=1)
        # Epoch,rp,ep,mse,mae,rmse,ssim,psnr,mae_99,rmse_99
        metrics_rmse.append(metrics[0][1:]) # skip first number
        rmse,mae = metrics_rmse[0][2]**0.5,metrics_rmse[0][3]
        
        # load mean, 99 and 01 for 1 var 
        irep,epoch = int(metrics_rmse[0][0]),int(metrics_rmse[0][1])
        out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_ep'+'_re'+ str(irep)+'_mk'+str(kmask)+'/'    
        filename99 = out_path + "sr_99th_epoch%d" % (epoch)+'.npz'
        filename01 = out_path + "sr_01th_epoch%d" % (epoch)+'.npz'
        filename_m = out_path + "sr_mean_epoch%d" % (epoch)+'.npz'   
        datald = np.load(filename99) # load
        rmse_99,mae_99 = datald['v2'],datald['v3']
        datald = np.load(filename01) # load
        rmse_01,mae_01 = datald['v2'],datald['v3']
        datald = np.load(filename_m) # load
        rmse_m,mae_m = datald['v2'],datald['v3']
        metrics_rmse8.append([mae,rmse,mae_m[0],rmse_m[0],mae_99[0],rmse_99[0],mae_01[0],rmse_01[0]])
    return metrics_rmse8,metrics_rmse

# load max rmse from sorted data from direct interpolation
def read_rmse_max_re(mod_name,kmask):
    mod_para=importlib.import_module(mod_name)
    opt = mod_para.opt
    rtra = mod_para.rtra
    ivar_hr = mod_para.ivar_hr
    out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+'_mask'+str(kmask)+'/'
    filename99m = out_path + 'hr_99per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
    filename01m = out_path + 'hr_01per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
    filename_mm = out_path + 'hr_mean_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
    datald = np.load(filename99m) # load
    rmse_99_re1,rmse_99_re2,rmse_99_re3,mae_99_re1,mae_99_re2,mae_99_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    datald = np.load(filename01m) # load
    rmse_01_re1,rmse_01_re2,rmse_01_re3,mae_01_re1,mae_01_re2,mae_01_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    datald = np.load(filename_mm) # load
    rmse_m_re1,rmse_m_re2,rmse_m_re3,mae_m_re1,mae_m_re2,mae_m_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
    # obtain rmse and mae 
    file_metric_intp = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
    metrics_re_bt = np.load(file_metric_intp,allow_pickle='TRUE').item()
    metrics_re = {}
    for key, value in metrics_re_bt.items():
        metrics_re[key] = sum(metrics_re_bt[key])/len(metrics_re_bt[key]) # / Nbatch_t
    rmse_re1,mae_re1 = metrics_re['mse_re1']**0.5,metrics_re['mae_re1']
    rmse_re2,mae_re2 = metrics_re['mse_re2']**0.5,metrics_re['mae_re2']
    rmse_re3,mae_re3 = metrics_re['mse_re3']**0.5,metrics_re['mae_re3']
    metrics_rmse8_re1 = np.stack((mae_re1,rmse_re1,mae_m_re1,rmse_m_re1,mae_99_re1,rmse_99_re1,mae_01_re1,rmse_01_re1))
    metrics_rmse8_re2 = np.stack((mae_re2,rmse_re2,mae_m_re2,rmse_m_re2,mae_99_re2,rmse_99_re2,mae_01_re2,rmse_01_re2))
    metrics_rmse8_re3 = np.stack((mae_re3,rmse_re3,mae_m_re3,rmse_m_re3,mae_99_re3,rmse_99_re3,mae_01_re3,rmse_01_re3))
    metrics_rmse8_re = np.stack((metrics_rmse8_re1,metrics_rmse8_re2,metrics_rmse8_re3))
    ichl0,ichl1 = ivar_hr[0]-3,ivar_hr[-1]-3+1
    return metrics_rmse8_re[:,:,ichl0:ichl1]

# load max rmse from sorted data from direct interpolation
def plot_bar(data,figname,axlab,leg,ticklab,figsize=[6,4],size_label=16,capt='(a)',
             leg_col=2,style='default'):
    # data(M,N) array: N bar at M xticks
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms    
    # from matplotlib import style 

    x = np.arange(data.shape[0])
    dx = (np.arange(data.shape[1])-data.shape[1]/2.)/(data.shape[1]+2.)
    d = 1./(data.shape[1]+2.)
    
    plt.style.context(style) # 'seaborn-deep', why not working
    plt.style.use(style) # 'seaborn-deep', why not working

    fig, ax=plt.subplots(figsize=figsize)
    for i in range(data.shape[1]):
        ax.bar(x+dx[i],data[:,i], width=d, label=leg[i], zorder=3)
    ax.set_xticks(np.arange(len(ticklab))) # for missing 1st tick
    ax.set_xticklabels(ticklab,rotation = 30,fontsize=size_label)
    ax.set_xlabel(axlab[0],fontsize=size_label)
    ax.set_ylabel(axlab[1],fontsize=size_label)
    ax.tick_params(axis="both", labelsize=size_label-1)
    ax.grid(zorder=0)
    plt.legend(ncol=leg_col,fontsize=size_label-2,handlelength=1.5,
               handleheight=0.4,labelspacing=0.3,columnspacing=1.0)  # framealpha=1
    if capt is not None: 
        trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
        plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    # plt.show()
    plt.close(fig) 


if __name__ == '__main__':
    
    epoc_num=[40,100]
    kmask = 1 
    mod_name= ['par01_s4','par01','par01_s32','par01_b2','par01_b16',
               'par11_s4','par11','par11_s32','par01_s64_','par11_s64_'] # for a single var
    metrics_rmse = []
    metrics_rmse8 = []
    metrics_rmse8_re = []
    for i in range(len(mod_name)):
        temp8,temp = read_rmse_max(mod_name[i],epoc_num,kmask)
        metrics_rmse.append(temp)
        metrics_rmse8.append(temp8)
        temp8_re = read_rmse_max_re(mod_name[i],kmask)
        metrics_rmse8_re.append(temp8_re)
        
    metrics_rmse = np.concatenate(metrics_rmse) # no 01,99,mean
    metrics_rmse8 = np.concatenate(metrics_rmse8) # global,mean,01,99 mae/rmse sr
    metrics_rmse8_re = np.concatenate(metrics_rmse8_re)
    metrics_rmse8_re = np.squeeze(metrics_rmse8_re)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae99','rmse99','mae01','rmse01']
    
    out_path = 'results_test/'
    ofname = 'matrics_sort_rmse_sr_mk'+str(kmask)+'_head.csv'
    data = np.vstack((ticklab,metrics_rmse8))
    df = pd.DataFrame(data.T,columns=['Test']+mod_name)
    df.to_csv(out_path + ofname, index=False)
    ofname = 'matrics_sort_rmse_sr_mk'+str(kmask)+'.csv'
    np.savetxt(out_path + ofname, np.array(metrics_rmse8).transpose(), delimiter=",")
    ofname = 'matrics_sort_rmse_re_mk'+str(kmask)+'.csv'
    np.savetxt(out_path + ofname, np.array(metrics_rmse8_re).transpose(), delimiter=",")    
    
# 2	84	0.0036245	0.0207491	0.0602038	0.990286	42.9561	0.0315584	0.0816812
# 4	97	0.00330294	0.0190089	0.0574712	0.991282	42.8812	0.0264529	0.0648147
# 0	92	0.00390554	0.0228462	0.0624943	0.99097	42.4772	0.0528953	0.0735867
# 4	94	0.0033776	0.0188265	0.0581171	0.991509	42.9092	0.0238448	0.0862082
# 2	97	0.00344729	0.0183907	0.0587137	0.991497	42.9137	0.0260512	0.0778656
# 2	100	0.000453581	0.0136775	0.0212974	0.994762	49.2086	0.0149868	0.0308653
# 4	100	0.000591102	0.0153468	0.0243126	0.995482	48.6883	0.0173125	0.0441447
# 1	99	0.00137558	0.0249506	0.0370888	0.991931	45.0416	0.0286035	0.0455947
# 4	90	0.00403554	0.0226314	0.0635259	0.991453	42.4885	0.0246024	0.0637689
# 3	91	0.00136153	0.0247601	0.0368989	0.992636	45.19	0.0222668	0.0439406


    figsize=[6,3.7]
    style='default' # 'default','seaborn-deep'
    
    # metrics_rmse8_re: (3*ncase,8,1), first dim, 2-bicubic, 1-bilinear,0-nearest
    label = ['res2','res6','res16','nearest','bilinear','bicubic']
    data = np.stack((metrics_rmse8[3,:],metrics_rmse8[1,:],metrics_rmse8[4,:],
                            metrics_rmse8_re[5,:],metrics_rmse8_re[4,:],metrics_rmse8_re[3,:]))
    data = data.T
    figname = 'compare_metrics_ssh_res.png'
    axlab = ['','Error in SSH (m)']
    plot_bar(data,figname,axlab,label,ticklab,figsize=figsize,size_label=16,capt='(a)',style=style)
    ofname = 'matrics_sort_rmse_mk'+str(kmask)+'_res.csv'
    np.savetxt(out_path + ofname, np.array(data), delimiter=",")    
    
    
    # # for scale 4-32
    # label = ['sr_s4','sr_s16','sr_s32','nearest_s4','bilinear_s4','bicubic_s4']
    # data = np.column_stack((metrics_rmse8[0:3,:].T,metrics_rmse8_re[2,:],
    #                        metrics_rmse8_re[1,:],metrics_rmse8_re[0,:]))
    # figname = 'compare_metrics_ssh_scale.png'
    # axlab = ['','Error in SSH (m)']
    # plot_bar(data,figname,axlab,label,ticklab,size_label=16,capt='(b)',style=style)

    # label = ['sr_s4','sr_s16','sr_s32','nearest_s4','bilinear_s4','bicubic_s4']
    # data = np.column_stack((metrics_rmse8[5:8,:].T,metrics_rmse8_re[17,:],
    #                        metrics_rmse8_re[16,:],metrics_rmse8_re[15,:]))
    # figname = 'compare_metrics_u_scale.png'
    # axlab = ['','Error in u (m/s)']
    # plot_bar(data,figname,axlab,label,ticklab,size_label=16,capt='(c)',style=style)

# for scale 4-64
    label = ['sr_s4','sr_s16','sr_s32','sr_s64','nearest_s4','bilinear_s4','bicubic_s4']
    data = np.column_stack((metrics_rmse8[0:3,:].T,metrics_rmse8[8,:],metrics_rmse8_re[2,:],
                           metrics_rmse8_re[1,:],metrics_rmse8_re[0,:])) # concatenate ,axis=1
    figname = 'compare_metrics_ssh_scale64.png'
    axlab = ['','Error in SSH (m)']
    plot_bar(data,figname,axlab,label,ticklab,figsize=figsize,size_label=16,capt='(b)',style=style)
    ofname = 'matrics_sort_rmse_mk'+str(kmask)+'_scale_ssh.csv'
    np.savetxt(out_path + ofname, np.array(data), delimiter=",")   

    data = np.column_stack((metrics_rmse8[5:8,:].T,metrics_rmse8[9,:],metrics_rmse8_re[17,:],
                           metrics_rmse8_re[16,:],metrics_rmse8_re[15,:]))
    figname = 'compare_metrics_u_scale64.png'
    axlab = ['','Error in u (m/s)']
    plot_bar(data,figname,axlab,label,ticklab,figsize=figsize,size_label=16,capt='(c)',style=style)
    ofname = 'matrics_sort_rmse_mk'+str(kmask)+'_scale_u.csv'
    np.savetxt(out_path + ofname, np.array(data), delimiter=",")   