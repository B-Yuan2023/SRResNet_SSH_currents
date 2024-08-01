#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:07:24 2023
functions to process nc files 
@author: g260218
"""
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
# import netCDF4
from datetime import date, datetime, timedelta
import math

# read data, interpolate and output
# nc_f: name of nc files
def read_cmemes_TG(nc_f,cri_QC=None,sshlim=None):
    nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
    nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)

    # Extract data from NetCDF file
    lon = float(nc_fid.geospatial_lon_min)
    lat = float(nc_fid.geospatial_lat_min)
    sid = nc_fid.id
    str_ts = nc_fid.time_coverage_start
    str_te = nc_fid.time_coverage_end
    
    ts = datetime.strptime(str_ts, '%Y-%m-%dT%H:%M:%SZ')#.date()
    te = datetime.strptime(str_te, '%Y-%m-%dT%H:%M:%SZ')#.date()

    time = nc_fid.variables['TIME'][:] # days refer to 1950.1.1
    dt = int((time[1]-time[0])*1440+0.5)
    tref = datetime(1950,1,1) # reference date in nc file 
    dtref = (datetime(1970,1,1)-datetime(1950,1,1)).days # reference in python 1970
    dcount = (ts - tref).days + 1
    
    sec = (time.data - dtref) * 86400.0 # seconds refer to 1970
    func = np.vectorize(datetime.utcfromtimestamp)
    timed= func(sec)
    # timed = timed.tolist()
    
    # time = np.ma.getdata(time)
    # sec = (time - dtref) * 86400.0 # seconds refer to 1970
    # timed = [datetime.fromtimestamp(sec[i]) for i in range(len(sec))]
    
    ssh = nc_fid.variables['SLEV'][:]  # shape is time, 1
    SLEV_QC = nc_fid.variables['SLEV_QC'][:]  # shape is time
    depth = nc_fid.variables['DEPH'][:]  # shape is time, 1
    DEPH_QC = nc_fid.variables['DEPH_QC'][:]  # shape is time
    ssh_max = ssh.max()
    # print(sid,'sshmax=',ssh_max)
    
    # remove possible bad values 
    # SLEV_QC: 0 "no_qc_performed; 1 good_data; 2 probably_good_data; 3 bad_data_that_are_potentially_correctable;  4 bad_data; 5 value_changed; 6 value_below_detection; 7 nominal_value; 8 interpolated_value; 9 missing_value" 
    if cri_QC is not None:
        ssh[SLEV_QC>cri_QC]=np.nan
    if sshlim is not None:
        ssh[ssh>sshlim[0]]=np.nan
        ssh[ssh<sshlim[1]]=np.nan
#     pref=infile.split('/')[-1].replace('.nc','')
#     fig = plt.figure()
#     plt.plot(timed,ssh,'k.')
#     plt.xlabel('time',fontsize=16)
#     plt.ylabel('ssh (m)',fontsize=16)
#     figname = pref+'ssh_.png'
#     plt.savefig(figname,dpi=100)
#     plt.close(fig)
#     plt.show()

    return sid,timed,ts,te,lon,lat,ssh,SLEV_QC

def plot_sites_var(timed,ssh,tlim=None,figname='Fig'):
    fig = plt.figure()
    plt.plot(timed,ssh,'k.')
    fig.autofmt_xdate()
    if tlim is not None:
        ax = plt.gca()
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    plt.xlabel('time',fontsize=16)
    plt.ylabel('ssh (m)',fontsize=16)
    plt.savefig(figname,dpi=100)
    plt.close(fig)
    plt.show()

def write_sites_info(sids,lons,lats,ts_all,te_all,outfile):
    data_all = np.rec.fromarrays((lons,lats,ts_all,te_all,sids))
    np.savetxt(outfile+'.dat', data_all, fmt='%f,%f,%s,%s,%s')
    #b = np.loadtxt(outfile+'.dat')    

def interp_var(lon_sta,lat_sta,lon,lat,var,method='max'):
    # lon_sta, lat_sta are longitude and latidude of 1 station
    # lon, lat are longitude and latidude of rectangular grids
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    nnodes = 4
    ix0 = int((lon_sta-lon[0])/dx)
    iy0 = int((lat_sta-lat[0])/dy)
    ix1 = ix0+1
    iy1 = iy0+1
    index = [[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]]
    assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
    ll_sta = [lon_sta,lat_sta]
    # ll_grd0 = [[lon[ix0],lat[iy0]],[lon[ix0],lat[iy1]],
    #           [lon[ix1],lat[iy1]],[lon[ix1],lat[iy0]]]  # clockwise
    # check if the grid has meaningful value 
    ll_grd = []
    var_use = []
    for k in range(nnodes):
        if var[index[k][0],index[k][1]] != 0:
            ll_grd.append([lon[index[k][0]],lat[index[k][1]]])
            var_use.append(var[index[k][0],index[k][1]])

    nnode_ = len(ll_grd)
    assert nnode_ > 0 , "locate on land!"
    
    weight = np.zeros(nnode_)
    dist = [] 
    for k in range(nnode_):
        dist.append(math.dist(ll_sta,ll_grd[k]))
        
    if method == 'max':
        ind_max = np.argmax(var_use)
        weight[ind_max] = 1.0
    elif method == 'nearest':
        ind_min = np.argmin(dist)
        weight[ind_min] = 1.0
    elif method == 'ave':
        weight = np.ones(nnodes)*1.0/nnode_
    elif method == 'idw':
        weight = 1/(dist+1e-10)/dist.sum()
    var_int = 0.0
    for k in range(nnode_):
        var_int += var_use[k]*weight[k]
    return var_int

def index_stations(lon_sta,lat_sta,lon,lat):
    # lon_sta, lat_sta are longitude and latidude of stations
    # lon, lat are longitude and latidude of rectangular grids
    nsta = len(lon_sta)
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    index = [None] * nsta
    for i in range(nsta):
        ix0 = int((lon_sta[i]-lon[0])/dx)
        iy0 = int((lat_sta[i]-lat[0])/dy)
        ix1 = ix0+1
        iy1 = iy0+1
        assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
        index[i] = np.array([[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]])
    index = np.vstack(index)
    return index

def index_weight_stations(lon_sta,lat_sta,lon,lat,method='nearest'):
    # lon_sta, lat_sta are longitude and latidude of stations
    # lon, lat are longitude and latidude of rectangular grids
    nsta = len(lon_sta)
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    dist = [None] * nsta
    index = [None] * nsta
    weight = [None] * nsta
    nnodes = 4
    for i in range(nsta):
        ix0 = int((lon_sta[i]-lon[0])/dx)
        iy0 = int((lat_sta[i]-lat[0])/dy)
        ix1 = ix0+1
        iy1 = iy0+1
        assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
        index[i] = [[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]]
        ll_sta = [lon_sta[i],lat_sta[i]]
        ll_grd = [[lon[ix0],lat[iy0]],[lon[ix0],lat[iy1]],
                  [lon[ix1],lat[iy1]],[lon[ix1],lat[iy0]]]  # clockwise
        dist[i] = []
        weight[i] = np.zeros(nnodes)
        for j in range(nnodes):
            dist[i].append(math.dist(ll_sta,ll_grd[j]))
            
        if method == 'nearest':
            ind_min = np.argmin(dist)
            weight[i][ind_min] = 1.0
        elif method == 'ave':
            weight[i] = np.ones(nnodes)*1.0/nnodes
        elif method == 'idw':
            weight[i] = 1/(dist[i]+1e-10)/dist[i].sum()
    return index, weight
        
