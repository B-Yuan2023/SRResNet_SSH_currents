#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 07:15:24 2024

pre-post functions for data

@author: g260218
"""

import numpy as np
from matplotlib import pyplot as plt
import netCDF4 

ntpd = 24 # number of time steps in an nc file


# functions for read variables in nc files 
def nc_load_all(nc_f,indt=None):
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
     # and create an instance of the ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))
    ssh = nc_fid.variables['elevation'][indt,:]  # shape is time, Ny*Nx
    uwind = nc_fid.variables['windSpeedX'][indt,:]  # shape is time, Ny*Nx
    vwind = nc_fid.variables['windSpeedY'][indt,:]  # shape is time, Ny*Nx
    swh = nc_fid.variables['sigWaveHeight'][indt,:]  # shape is time, Ny*Nx
    pwp = nc_fid.variables['peakPeriod'][indt,:]  # shape is time, Ny*Nx
    ud = nc_fid.variables['depthAverageVelX'][indt,:]  # shape is time, Ny*Nx
    vd = nc_fid.variables['depthAverageVelY'][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    
    mask = np.ma.getmask(ssh)
    
    FillValue=0.0 # np.nan
    ssh = ssh.filled(fill_value=FillValue)
    uwind = uwind.filled(fill_value=FillValue)
    vwind = vwind.filled(fill_value=FillValue)
    swh = swh.filled(fill_value=FillValue)
    pwp = pwp.filled(fill_value=FillValue)
    ud = ud.filled(fill_value=FillValue)
    vd = vd.filled(fill_value=FillValue)
    
    ssh = np.ma.getdata(ssh) # data of masked array
    uw = np.ma.getdata(uwind) # data of masked array
    vw = np.ma.getdata(vwind) # data of masked array
    swh = np.ma.getdata(swh) # data of masked array
    pwp = np.ma.getdata(pwp) # data of masked array
    ud = np.ma.getdata(ud) # data of masked array
    vd = np.ma.getdata(vd) # data of masked array
    return time,lon,lat,ssh,ud,vd,uw,vw,swh,pwp,mask


def nc_load_depth(nc_f):
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
    # nc_fid.variables.keys() # list(nc_fid.variables)
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    depth = nc_fid.variables['depth'] # shape is Ny*Nx
    mask = np.ma.getmask(depth)
    # FillValue=0.0 # np.nan
    # depth = depth.filled(fill_value=FillValue)
    lon = np.ma.getdata(lon) # data of masked array
    lat = np.ma.getdata(lat) # data of masked array
    depth = np.ma.getdata(depth) # data of masked array
    return depth,lon,lat,mask


# instance or dataset normalization
# ivar = [3,4,5] # ssh, ud, vd
def nc_var_normalize(nc_f,indt,ivar,varmaxmin=None):
    nvar = len(ivar)
    Nx = len(nc_load_all(nc_f,indt)[1])
    Ny = len(nc_load_all(nc_f,indt)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        var = nc_load_all(nc_f,indt)[ivar[i]]
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flipud(var) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
                
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
        #data = np.array(data).reshape(data.shape[0],data.shape[1],1) # height, width, channel (top to bot)
    # data = np.dstack(data)
    # if nvar==1:
    #     data = np.repeat(data[..., np.newaxis], 3, -1)  # make 1 channel to 3 channels for later interpolation and trained model like vgg19
    return data 

# denormalize 
def var_denormalize(var,varmaxmin):
    # var(N,C,H,W)
    nc = var.shape[1]
    var = np.flip(var,2) # flip the dimenson of height as in narmalize
    data = np.zeros(shape=var.shape)
    for i in range(nc):
        vmax = varmaxmin[i,0]
        vmin = varmaxmin[i,1]
        data[:,i,:,:] = var[:,i,:,:]*(vmax-vmin) + vmin 
    return data 


# sorted ssh in hour
def find_max_global(files, ivar=[3]):
    nfile = len(files)
    nvar = len(ivar)
    ind_sort = [[]]*nvar
    var_sort = [[]]*nvar
    # ind_file = [[]]*nvar
    # ind_it = [[]]*nvar
    for i in range(nvar):
        var_comb = []
        # var_file = []
        # var_it = []
        for indf in range(nfile):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            var_max = var.max(axis=(1,2)) # maximum in 2d space, note during ebb sl can be <0
            for indt in range(ntpd):
                var_comb.append(var_max[indt])
                # var_file.append(indf) # the indf th file, not file name index
                # var_it.append(indt)
        ind_sort[i] = sorted(range(len(var_comb)), key=lambda k: var_comb[k], reverse=True)
        var_sort[i] = [var_comb[k] for k in ind_sort[i]]
        # var_sort[i] = sorted(var_comb)
        # ind_file[i] = [var_file[k] for k in ind_sort[i]]
        # ind_it[i] = [var_it[k] for k in ind_sort[i]]
    return var_sort,ind_sort #,ind_file,ind_it


def plt_sub(sample,ncol,figname,ichan=0,clim=[0,1],cmp='bwr',contl=None,txt=None,loc_txt=None,size_txt=10):  # normalized sample(nk,1,ny,nx)
    nsub = len(sample)
    columns = ncol
    rows = int(-(-(nsub/columns))//1)
    fig = plt.figure()
    for i in range(0,nsub):
        fig.add_subplot(rows, columns, i+1)        
        plt.imshow(sample[i,ichan,:,:],cmap=cmp) # bwr,coolwarm
        plt.axis('off')
        plt.clim(clim[0],clim[1]) 
        plt.tight_layout()
        #plt.title("First")
        if contl is not None: # add 0 contour
            plt.contour(sample[i,ichan,:,:], levels=contl, colors='black', linewidths=1)
        if txt is not None: 
            # size_txt = 12
            ax = plt.gca()
            for j in range(len(loc_txt)):
                plt.text(loc_txt[j][0],loc_txt[j][1], txt[j],fontsize=size_txt,ha='left', va='top', transform=ax.transAxes) #add text
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    
# def plt_contour_list(lat,lon,sample,figname,lev=11,cmap='bwr',clim=None,unit=None,title=None):  # sample is a list with k array[C,nx,ny]
#     nsub = len(sample)
#     fig, axes = plt.subplots(1, nsub, figsize=(5 * nsub, 5))
#     for i in range(0,nsub):
#         ax = axes[i] if nsub > 1 else axes
#         # ax.set_facecolor('xkcd:gray')
#         if clim:
#             vmin, vmax = clim[i]
#             cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
#             # cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
#         else:
#             cf = ax.contourf(lat,lon,sample[i],levels=lev,cmap=cmap) # bwr,coolwarm
#         cbar = fig.colorbar(cf, ax=ax)
#         ax.set_title(title[i] if title else f'Array {i + 1}')
#         if unit:
#             cbar.set_label(unit[i])
#         ax.set_xlabel('lon',fontsize=16)
#         ax.set_ylabel('lat',fontsize=16)
#         plt.tight_layout()
#     plt.savefig(figname,dpi=100) #,dpi=100    
#     plt.close(fig)
    
    
def plt_contour_list(lat,lon,sample,figname,lev=20,subsize = [5,4],cmap='bwr',clim=None,unit=None,
                    title=None,nrow=1,axoff=0,capt=None,txt=None,loc_txt=None):  # sample is a list with k array[C,nx,ny]
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18
    axes = axes.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axes[irm_ax[i]])
    for i in range(0,nsub):
        ax = axes[i] if nsub > 1 else axes
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
        else:
            cf = ax.contourf(lat,lon,sample[i],levels=lev,cmap=cmap) # bwr,coolwarm
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=size_tick)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)
        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None: 
            plt.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            plt.text(0.00, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)    
    
    
def plt_pcolor_list(lat,lon,sample,figname,subsize = [5,4],cmap='bwr',clim=None,unit=None,
                    title=None,nrow=1,axoff=0,capt=None,txt=None,loc_txt=None):  # sample is a list with k array[C,nx,ny]
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18
    axes = axes.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axes[irm_ax[i]])
    for i in range(0,nsub):
        ax = axes[i] if nsub > 1 else axes
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.pcolor(lat, lon, sample[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            cf = ax.pcolor(lat, lon, sample[i], cmap=cmap)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=size_tick)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)
        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None: 
            plt.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            plt.text(0.00, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    

def plot_sites_cmpn(time_lst,dat_lst,tlim=None,figname='Fig',axlab=None,leg=None,
                   leg_col=1, legloc=None,line_sty=None,style='default',capt=''):
    import matplotlib.transforms as mtransforms    
    
    size_tick = 14
    size_label = 16
    # size_title = 18
    fig = plt.figure()
    ndat = len(time_lst)
    # line_sty=['k','b','r','m','g','c']
    with plt.style.context(style):
        for i in range(ndat): 
            if line_sty is not None and len(line_sty)>=ndat:
                plt.plot(time_lst[i],dat_lst[i],line_sty[i]) # ,mfc='none'
            else:
                plt.plot(time_lst[i],dat_lst[i])
    fig.autofmt_xdate()
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    ax.tick_params(axis="both", labelsize=size_tick)     
    if tlim is not None:
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=size_label)
        plt.ylabel(axlab[1],fontsize=size_label)
    if leg is None:
        leg = [str(i) for i in range(ndat)]
    else:
        leg = leg[:ndat]
    plt.tight_layout()
    if legloc is None:
        plt.legend(leg,ncol=leg_col,fontsize=size_tick)
    else: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C
        plt.legend(leg,ncol=leg_col,fontsize=size_tick,loc=2,bbox_to_anchor=legloc)    
    plt.savefig(figname,dpi=150)
    plt.close(fig)
    plt.show()
    

def plot_sites_cmp(time_TG,ssh_TG,time,ssh,tlim=None,figname=None,axlab=None,leg=None):
    fig = plt.figure()
    plt.plot(time_TG,ssh_TG,'k.')
    plt.plot(time,ssh,'b')
    fig.autofmt_xdate()
    if tlim is not None:
        ax = plt.gca()
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=14)
        plt.ylabel(axlab[1],fontsize=14)
    if leg is None:
        leg = ['ref','mod']
    if figname is None:
        figname = 'Fig'
    plt.legend(leg)     
    plt.savefig(figname,dpi=100)
    plt.close(fig)
    plt.show()        
    
    
def plot_mod_vs_obs(mod,obs,figname,axlab=('Target','Mod',''),leg=None,alpha=0.5,
                    marker='o',figsize=(4,4),fontsize=16,capt=''):
    """
    Plot model data against observation data to visualize bias in the model.
    Parameters:
        mod (list of arrays): Model data to be plotted.
        obs (array-like): Observation data to be plotted.
        label (tuple, optional): Labels for x-axis, y-axis, and title.
    """
    import matplotlib.transforms as mtransforms    

    fig= plt.figure(figsize=figsize)
    # plt.style.use('seaborn-deep')
    # if leg is None:
    #     leg = [str(i) for i in range(len(mod))]
    if len(marker) < len(mod):
        marker = ['o' for i in range(len(mod))]
    # Plot the scatter plot
    for i in range(len(mod)):
        if leg is not None:
            plt.scatter(obs, mod[i], alpha=0.3, marker=marker[i],label=leg[i]) # marker=marker, color='blue',
        else:
            plt.scatter(obs, mod[i], alpha=0.3, marker=marker[i]) # marker=marker, color='blue',

    # Set the same limits for x and y axes
    max_val = max(np.nanmax(obs), np.nanmax(np.array(mod)))
    min_val = min(np.nanmin(obs), np.nanmin(np.array(mod)))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)    
    # Set the same ticks for x and y axes
    ticks = plt.xticks()
    plt.xticks(ticks[0],fontsize=fontsize-2)
    plt.yticks(ticks[0],fontsize=fontsize-2)
    
    # Plot the perfect fit line (y = x)
    plt.plot(ticks[0], ticks[0], linestyle='dashed', color='black') 

    plt.xlabel(axlab[0], fontsize=fontsize)
    plt.ylabel(axlab[1], fontsize=fontsize)
    plt.title(axlab[2], fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    plt.grid(True)
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig(figname,dpi=100) #,dpi=100    
    plt.close(fig)
    
def plot_distri(data,figname,bins=10, axlab=('Val','P',''),leg=('1', ), 
                   figsize=(8, 6), fontsize=12,capt='',style='default'):
    """
    Compare the distribution of data using histograms.
    Parameters:
        data (list of arrays with same length): data, one array corresponds to 1 histogram.
        bins (int or sequence, optional): Number of bins or bin edges. Default is 10.
"""
    from matplotlib.ticker import PercentFormatter 
    import matplotlib.transforms as mtransforms    
    # from matplotlib import style 
    fig = plt.figure(figsize=figsize)
    # plt.style.use(style) #'seaborn-deep'
    plt.style.context(style)
    # Calculate the bin edges
    xmin = min([np.nanmin(np.array(data[i])) for i in range(len(data))])
    xmax = max([np.nanmax(np.array(data[i])) for i in range(len(data))])
    hist_range = (xmin, xmax)
    bins = np.linspace(hist_range[0], hist_range[1], bins+1)

    # Plot histogram for observation data
    # for i in range(length(data)):
    # plt.hist(data[i], bins=bins, color=color[i], alpha=0.5, label=label[i]) # , align='left'

    # to plot the histogam side by side
    weights=[np.ones(len(data[i])) / len(data[i]) for i in range(len(data))]
    plt.hist(data,bins=bins,weights=weights, alpha=0.5, label=leg) # , align='right'

    plt.xlabel(axlab[0], fontsize=fontsize)
    plt.ylabel(axlab[1], fontsize=fontsize)
    plt.title(axlab[2], fontsize=fontsize)
    plt.legend(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # for array of the same length
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(data[0]))) # for array of the same length

    plt.grid(True)
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    plt.tight_layout()
    # plt.show()
    plt.savefig(figname,dpi=300) #,dpi=100    
    plt.close(fig)