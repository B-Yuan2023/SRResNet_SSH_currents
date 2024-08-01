"""
statistic analysis of reconstruted test data

"""

import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorResNet
from datasets import myDataset,find_maxmin_global
from funs_prepost import (var_denormalize,nc_load_all,ntpd,plot_mod_vs_obs,
                          plot_distri)

import torch
import pandas as pd

import sys
import importlib
mod_name= 'par11'          #'par04' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

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
    rep = [4]
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    capt_dist = '(a)'
    style = 'seaborn-deep' #  plot style 
    epoc_num = [100]
    # epoc_num = np.arange(40,opt.N_epochs+1)
    key_ep_sort = 0 # to use epoc here or load sorted epoc no. 
    nepoc = 1 # no. of sorted epochs for analysis

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    train_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='train',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr) #,ind_sort=ind_sort

    test_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='val',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr) # rtra = rtra,
    
    opath_st = 'statistics' + suf+'_mk'+str(kmask) +'/'
    if not os.path.exists(opath_st):
        os.makedirs(opath_st)

    # data_train = DataLoader(
    #     train_set,
    #     batch_size=opt.batch_size,
    #     num_workers=opt.n_cpu,
    # )
    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=1, # opt.n_cpu
    )        
    
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    # get logitude and latitude of data 
    nc_f = test_set.files_hr[0]
    lon = nc_load_all(nc_f,0)[1]
    lat = nc_load_all(nc_f,0)[2]
    # mask = nc_load_all(nc_f,0)[10] # original data
    
    # identify the point with max/min value in the dataset
    varm_hr_test,ind_varm,file_varm = find_maxmin_global(test_set.files_hr, ivar_hr)
    temp = np.unravel_index(ind_varm.flatten(), (ntpd,len(lat),len(lon)))
    index_pnt = np.array([temp[1],temp[2]]).transpose() # y,x index [Nvar,2]
    ll_sta = np.array([lon[index_pnt[:,1]],lat[index_pnt[:,0]]]).transpose()
    
    in_path = 'results_test/'+'SRF_'+str(opt.up_factor)+'_mask'+str(kmask)+'/' # hr_all, etc
    
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)']
    
    out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_st'+'_mk'+str(kmask)+'/'
    os.makedirs(out_path, exist_ok=True)

    # layers: repeat/epoch/batch/channel 
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')

        # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
        ipath_nn = 'nn_models_' + str(opt.up_factor) + suf +'/' # 
    
        # Initialize generator and discriminator
        generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                    n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor).eval()

        # metrics = {'ep':[],'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], 'mae_99': [],'rmse_99': [],} # eopch mean
        # metrics_chl = {}
        
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_test_metrics.npy'
        # np.save(opath_st + os.sep + ofname, metrics) 
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_test_metrics_bt.npy'
        # np.save(opath_st + os.sep + ofname, metrics_bt) 
        # metrics_bt = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        
        if key_ep_sort:
            # choose sorted epoc number that gives the smallest rmse
            ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort_rmse.csv' #  rank based on 
            # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort'+tstr+'.csv' # rank based on rt_use highest ssh/
            ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
            epoc_num = ep_sort.flatten()[0:nepoc*nchl_o] 
            epoc_num = list(set(epoc_num.tolist())) # remove duplicates,order not reseved
    
        for epoch in epoc_num:
            start = time.time() 
            model_name = 'netG_epoch_%d_re%d.pth' % (epoch,irep)
            if cuda:
                generator = generator.cuda()
                checkpointG = torch.load(ipath_nn + model_name)
            else:
                checkpointG = torch.load(ipath_nn + model_name, map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            
            sr_all = []
            # hr_all = []
            # hr_restore1_all = []
            # hr_restore2_all = []
            # hr_restore3_all = []

# reconstruct all test data
            for i, dat in enumerate(data_test):                
                
                dat_lr = Variable(dat["lr"].type(Tensor))
                dat_hr = Variable(dat["hr"].type(Tensor))
                gen_hr = generator(dat_lr)
                sr_norm0 = var_denormalize(gen_hr.detach().cpu().numpy(),varm_hr) # (N,C,H,W), flipud height back
                # hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
                
                if kmask == 1:                     
                # get mask for time step, one batch may cover muliple files
                    # for ib in range(0,opt.batch_size):
                    # ib = int(opt.batch_size/2) # use mask of the middle sample to represent the mask of this batch
                    mask = sr_norm0==sr_norm0 # initialize the boolean array with the shape of hr_norm0
                    for ib in range(opt.batch_size):  # use mask for each sample/time
                        it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
                        indf = int(it/ntpd)  # id of ncfile 
                        indt = it % ntpd  # the i th time step in a ncfile
                        nc_f = test_set.files_hr[indf]
                        for ichl in range(nchl):
                            mask[ib,ichl,:,:] = nc_load_all(nc_f,indt)[10] # mask at 1 time in a batch
                    sr_norm0[mask] = np.nan 
                    # hr_norm0[:,:,mask] = np.nan
                
                sr_all.append(sr_norm0)
                # hr_all.append(hr_norm0)

            # hr_all = np.array(hr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
            sr_all = np.array(sr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
            end = time.time()
            elapsed = (end - start)
            print('cost ' + str(elapsed) + 's')

            # output  
            for i in range(nchl):
                ichl = ivar_hr[i]-3
                # load saved reconstruced hr (using cal_metrics_intp.py)
                filename = in_path + 'c%d_'%(ichl)+'hr_all'+'_train%4.2f'%(rtra)+'.npz'
                datald = np.load(filename) # load
                hr_all = datald['v0'] #[N,H,W]
                filename = in_path + 'c%d_'%(ichl)+'hr_restore_all'+'_train%4.2f'%(rtra)+'.npz'
                datald = np.load(filename) # load
                hr_restore1_all = datald['v0'][0] # bibubic 
                hr_restore2_all = datald['v0'][1] # bilinear
                hr_restore3_all = datald['v0'][2] # nearest
                
                unit_var = unit_suv[ichl]
                
                # # plot reconstructed vs target, select one point maximum
                # axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                # leg = ['sr','nearest','bilinear','bicubic']
                # target = hr_all[:,index_pnt[i]]
                # var = [sr_all[:,i,index_pnt[i]],
                #         hr_restore3_all[:,index_pnt[i]],
                #         hr_restore2_all[:,index_pnt[i]],
                #         hr_restore1_all[:,index_pnt[i]],
                #         ]
                # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_max_pnt'+'_re'+ str(irep)+'_intp3.png'
                # plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=leg,alpha=0.3,
                #                 figsize=(5, 5), fontsize=16,capt='(b)')
                
                # plot reconstructed vs target, select one point maximum
                axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                target = hr_all[:,index_pnt[2*i]] # index of max 2i
                var = sr_all[:,i,index_pnt[2*i]],
                figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_max_pnt'+'_re'+ str(irep)+'.png'
                plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=None,alpha=0.3,
                                figsize=(5, 5), fontsize=16,capt='(b)')

                # plot reconstructed vs target, select one point minimum
                axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                target = hr_all[:,index_pnt[2*i+1]] # index of min 2i+1
                var = sr_all[:,i,index_pnt[2*i+1]],
                figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_min_pnt'+'_re'+ str(irep)+'.png'
                plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=None,alpha=0.3,
                                figsize=(5, 5), fontsize=16,capt='(c)')           
                
                # # test a random point in the domain
                # ix = np.arange(8, len(lat)-1, 16) #  non-shared poins by lr & hr, initial index links to scale
                # iy = np.arange(8, len(lon)-1, 16)
                # xv, yv = np.meshgrid(ix, iy)
                # index_pnts = np.vstack((np.int_(xv.flatten()), np.int_(yv.flatten()))).T
                # ll_stas = np.array([lat[index_pnts[:,0]],lon[index_pnts[:,1]]]).transpose() # should corresponds to (H,W), H[0]-lowest lat
                # for ip in range(len(ll_stas)):
                #     axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                #     target = hr_all[:,index_pnts[ip]] # index of min 2i+1
                #     var = sr_all[:,i,index_pnts[ip]],
                #     figname = out_path+"c%d_epoch%d_re%d" % (ichl,epoch,irep) +'_ll%4.3f_%4.3f'%(ll_stas[ip,1],ll_stas[ip,0])+'.png'
                #     plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=None,alpha=0.3,
                #                     figsize=(5, 5), fontsize=16) # ,capt='(b)'
                

                # plot distribution of reconstructed vs target, all data, histogram
                axlab = (unit_var,'Frequency','')
                leg = ['hr','sr','bilinear'] #,'nearest'
                target = hr_all.flatten()
                mask1d = ~np.isnan(target)                
                var = [target[mask1d],
                       sr_all[:,i,:,:].flatten()[mask1d],
                       hr_restore2_all.flatten()[mask1d],
                       # hr_restore3_all.flatten()[mask1d],
                       ]
                figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_dist_all'+'_re'+ str(irep)+'.png'
                plot_distri(var,figname,bins=20,axlab=axlab,leg=leg,
                               figsize=(5, 5), fontsize=16,capt=capt_dist,style=style) # ,capt='(a)'
                

                # plot distribution of reconstructed vs target, in mini-region, histogram
                nx = 2 # number of region in x directions
                ny = 2 # number of region in y directions
                nsub = nx*ny
                npy,npx = int(hr_shape[0]/ny), int(hr_shape[0]/nx) # no. of points in subregion
                for iy in range(ny):
                    iy0,iy1 = npy*iy,npy*(iy+1)
                    for ix in range(nx):
                        ix0,ix1 = npx*ix,npx*(ix+1)
                        target = hr_all[:,iy0:iy1,ix0:ix1].flatten()
                        mask1d = ~np.isnan(target)                
                        var = [target[mask1d],
                               sr_all[:,i,iy0:iy1,ix0:ix1].flatten()[mask1d],
                               hr_restore2_all[:,iy0:iy1,ix0:ix1].flatten()[mask1d],
                               # hr_restore3_all.flatten()[mask1d],
                               ]
                        figname = out_path+"c%d_ep%d_re%d" % (ichl,epoch,irep) +'_dist_sub%d_%d_%d'%(nsub,ix,iy)+'.png'
                        if len(var[0])>1:
                            plot_distri(var,figname,bins=20,axlab=axlab,leg=leg,
                                    figsize=(5, 5), fontsize=16,capt=capt_dist,style=style) # ,capt='(a)'
                
                # plot reconstructed vs target, maximum of each sample, not 1 vs 1
                # axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                # leg = ['sr','nearest','bilinear','bicubic']
                # target = np.nanmax(hr_all,axis=(1,2)) 
                # var = [np.nanmax(sr_all[:,i,:,:],axis=(1,2)),
                #        np.nanmax(hr_restore3_all,axis=(1,2)),
                #        np.nanmax(hr_restore2_all,axis=(1,2)),
                #        np.nanmax(hr_restore1_all,axis=(1,2)),
                #        ]
                # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_max_sample'+'_re'+ str(irep)+'_intp3.png'
                # plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=leg,alpha=0.3,
                #                 figsize=(5, 5), fontsize=16,capt='(b)')

                # # plot reconstructed vs target, maximum of each sample
                # axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                # leg = ['sr','bilinear','nearest']
                # target = np.nanmax(hr_all,axis=(1,2)) 
                # var = [np.nanmax(sr_all[:,i,:,:],axis=(1,2)),
                #        np.nanmax(hr_restore2_all,axis=(1,2)),
                #        # np.nanmax(hr_restore3_all,axis=(1,2)),
                #        ]
                # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_max_sample'+'_re'+ str(irep)+'.png'
                # plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=leg,alpha=0.3,
                #                 figsize=(5, 5), fontsize=16,capt='(b)')

                # # plot reconstructed vs target, maximum in time for each point, too large dataset
                # axlab = ('Target '+unit_var,'Reconstructed '+unit_var,'')
                # leg = ['sr','bilinear'] #,'nearest'
                # target = np.nanmax(hr_all,axis=(0)).flatten()
                # mask1d = ~np.isnan(target)
                # target = target[mask1d]
                # var = [np.nanmax(sr_all[:,i,:,:],axis=(0)).flatten()[mask1d],
                #        np.nanmax(hr_restore2_all,axis=(0)).flatten()[mask1d],
                #        # np.nanmax(hr_restore3_all,axis=(0)).flatten()[mask1d],
                #        ]
                # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_max_time'+'_re'+ str(irep)+'.png'
                # plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=leg,alpha=0.3,
                #                 figsize=(5, 5), fontsize=16,capt='(c)')

                # plot reconstructed vs target, all data







