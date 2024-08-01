#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:27:02 2024

sort the files with the maximum of variables 
@author: g260218
"""

import numpy as np
import os
import glob
from funs_prepost import (find_max_global,nc_load_all)

from datetime import datetime # , date, timedelta

def lst_flatten(xss):
    return [x for xs in xss for x in xs]

# sorted ssh per file (day)
def find_max_global_file(files, ivar=[3]):
    nfile = len(files)
    nvar = len(ivar)
    ind_sort = [[]]*nvar
    var_sort = [[]]*nvar
    for i in range(nvar):
        var_comb = []
        for indf in range(nfile):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            var_max = var.max() # maximum in time dim 0 and space dim 1,2
            var_comb.append(var_max)
        ind_sort[i] = sorted(range(len(var_comb)), key=lambda k: var_comb[k]) # , reverse=True, increasing
        var_sort[i] = [var_comb[k] for k in ind_sort[i]]
    return var_sort,ind_sort 


import importlib
mod_name= 'par11_s64_'  # par01:ssh, par11:u, par21:v, 
mod_para=importlib.import_module(mod_name)

if __name__ == '__main__':
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
    
    opath_st = 'statistics' + suf +'/'
    opath_st_hr = 'statistics_hr'+'_%d_%d'%(opt.hr_height, opt.hr_width)+'/' 
    if not os.path.exists(opath_st_hr):
        os.makedirs(opath_st_hr)
    
    dateref = datetime(2017,1,2) # out2d_interp_001.nc corresponds to 2017.1.2
    ntpd = int(24) 
    
    # use the time with the largest ssh 
    files_lr = sorted(glob.glob(dir_lr + "/*.nc"))
    files_hr = sorted(glob.glob(dir_hr + "/*.nc"))
    nfile = len(files_hr)
    ichlo = ivar_hr[0]-3 # only work for one variable
    
    # estimate maximum var in file in train/test set, test for single var
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_file'+dir_hr.split("_",1)[1]+'.npz'
    if not os.path.isfile(filename):
        var_sort,ind_sort = find_max_global_file(files_hr, ivar=[ivar_hr[0]]) # find maximum index 
        np.savez(filename, v1=var_sort,v2=ind_sort) 
        ofname = 'var%d'%ichlo+'_sorted_file'+dir_hr.split("_",1)[1]+'.csv'
        combined_ind= np.column_stack((np.array(var_sort[0]), np.array(ind_sort[0])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
    else:
        datald = np.load(filename) # load
        nfl = datald['v1'].size
        var_sort = datald['v1']
        ind_sort = datald['v2']

    
    rt_use = 0.01 # use top largest values of testing data for testing 
    tstr = '_rk%4.2f'%rt_use
    
    kfsort = 0 # to sort the file first in desending order, next sort in tran/test
    if kfsort==1: 
        ind_train = np.arange(0,int(nfile*rtra))    # 
        ind_valid= np.delete(np.arange(0,nfile),ind_train)
        nt_test = len(ind_valid)*ntpd
        nt_train = len(ind_train)*ntpd
        files_lr = [files_lr[i] for i in ind_sort[0]] 
        files_hr = [files_hr[i] for i in ind_sort[0]]  # files with var from small to large
        files_lr_test = [files_lr[i] for i in ind_valid]
        files_hr_test = [files_hr[i] for i in ind_valid]
        suf = '_fs'
    else:
        ind_train = np.arange(15,int(nfile*rtra)+15)    # exclude max
        ind_valid= np.delete(np.arange(0,nfile),ind_train)
        nt_test = len(ind_valid)*ntpd
        nt_train = len(ind_train)*ntpd
        files_lr_test = [files_lr[i] for i in ind_valid]
        files_hr_test = [files_hr[i] for i in ind_valid]
        suf = ''

    # estimate maximum ssh  in hour in train/test set, files in order
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_train'+dir_hr.split("_",1)[1]+'_rt%4.2f'%(rtra)+suf+'.npz'
    if not os.path.isfile(filename):
        files_hr_train = [files_hr[i] for i in ind_train]
        var_sort_train,ind_sort_train = find_max_global(files_hr_train, ivar=[ivar_hr[0]]) # find maximum index 
        np.savez(filename, v1=var_sort_train,v2=ind_sort_train) 
        ofname = 'var%d'%ichlo+'_sorted_train'+dir_hr.split("_",1)[1]+'_rt%4.2f'%(rtra)+tstr+suf+'.csv'
        combined_ind= np.column_stack((np.array(var_sort_train[0][0:int(nt_train*rt_use)]), np.array(ind_sort_train[0][0:int(nt_train*rt_use)])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
    
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_train'+dir_lr.split("_",1)[1]+'_rt%4.2f'%(rtra)+suf+'.npz'
    if not os.path.isfile(filename):
        files_lr_train = [files_lr[i] for i in ind_train]
        var_sort_train,ind_sort_train = find_max_global(files_lr_train, ivar=[ivar_hr[0]]) # find maximum index 
        np.savez(filename, v1=var_sort_train,v2=ind_sort_train) 
        ofname = 'var%d'%ichlo+'_sorted_train'+dir_lr.split("_",1)[1]+'_rt%4.2f'%(rtra)+tstr+suf+'.csv'
        combined_ind= np.column_stack((np.array(var_sort_train[0][0:int(nt_train*rt_use)]), np.array(ind_sort_train[0][0:int(nt_train*rt_use)])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
        
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_test'+dir_lr.split("_",1)[1]+'_rt%4.2f'%(rtra)+suf+'.npz'
    if not os.path.isfile(filename):
        var_sort_test,ind_sort_test = find_max_global(files_lr_test, ivar=[ivar_hr[0]]) # find maximum index 
        np.savez(filename, v1=var_sort_test,v2=ind_sort_test)
        ofname = 'var%d'%ichlo+'_sorted_test'+dir_lr.split("_",1)[1]+'_rt%4.2f'%(rtra)+tstr+suf+'.csv'
        combined_ind= np.column_stack((np.array(var_sort_test[0][0:int(nt_test*rt_use)]), np.array(ind_sort_test[0][0:int(nt_test*rt_use)])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
        
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_test'+dir_hr.split("_",1)[1]+'_rt%4.2f'%(rtra)+suf+'.npz'
    if not os.path.isfile(filename):
        var_sort,ind_sort = find_max_global(files_hr_test, ivar=[ivar_hr[0]]) # find maximum index 
        np.savez(filename, v1=var_sort,v2=ind_sort) 
        ofname = 'var%d'%ichlo+'_sorted_test'+dir_hr.split("_",1)[1]+'_rt%4.2f'%(rtra)+tstr+suf+'.csv'
        combined_ind= np.column_stack((np.array(var_sort[0][0:int(nt_test*rt_use)]), np.array(ind_sort[0][0:int(nt_test*rt_use)])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
        # var_sort = var_sort[0:int(nt_test*rt_use)]
        # ind_sort = ind_sort[0:int(nt_test*rt_use)]  # index of maximum ssh 
        
        