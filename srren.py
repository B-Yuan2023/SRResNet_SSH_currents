"""
Super-resolution Residual Network.

"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorResNet 
from datasets import myDataset,my_loss
from funs_prepost import plt_sub

import torch
from pytorch_msssim import ssim as ssim_torch
from math import log10
import pandas as pd

import sys
import importlib
mod_name= sys.argv[1] #'par01' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 

if mod_para.krand == 0:     # no randomness
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    suf0 = mod_para.suf0
    rtra = mod_para.rtra
    ivar_lr = mod_para.ivar_lr
    ivar_hr = mod_para.ivar_hr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(ivar_lr)
    nchl_o = len(ivar_hr)
    nrep = mod_para.nrep
    rep = list(range(0,nrep))
    if hasattr(mod_para, 'ind_sort'):
        ind_sort = mod_para.ind_sort # index of sorted files with increasing var
    else:
        ind_sort = None
    # rep = [3,4]
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    hr_shape = (opt.hr_height, opt.hr_width)

    train_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='train',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ind_sort=ind_sort)
    test_set = myDataset(opt.dir_lr,opt.dir_hr, hr_shape=hr_shape, up_factor=opt.up_factor,
                          mode='val',rtra = rtra,ivar_lr=ivar_lr,ivar_hr=ivar_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ind_sort=ind_sort)
    for irep in rep:
        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        data_train = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        data_test = DataLoader(
            test_set,
            batch_size=opt.batch_size, 
            num_workers=opt.n_cpu,
        )        
        Nbatch = len(data_train)
        Nbatch_t = len(data_test)
    
        out_path = 'results_training/SRF_' + str(opt.up_factor) + suf + '_re'+str(irep)+'/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    
        # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
        ipath_nn = 'nn_models_' + str(opt.up_factor) + suf0 +'/' # 
            
        opath_nn = 'nn_models_' + str(opt.up_factor) + suf +'/' # 
        if not os.path.exists(opath_nn):
            os.makedirs(opath_nn)
    
        opath_st = 'statistics' + suf +'/'
        if not os.path.exists(opath_st):
            os.makedirs(opath_st)
    
        cuda = torch.cuda.is_available()
        
        
        # Initialize generator and discriminator
        generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                    n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor)
        
        if cuda:
            generator = generator.cuda()
        
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
        if opt.n0_epoch != 0:
            # Load pretrained models
            if torch.cuda.is_available():
                checkpointG = torch.load(ipath_nn+'netG_epoch_%d.pth' % (opt.n0_epoch))
            else:
                checkpointG = torch.load(ipath_nn+'netG_epoch_%d.pth' % (opt.n0_epoch), map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        
        # ----------
        #  Training
        # ----------
        results = {'loss_G': [], 'psnr': [], 'ssim': [], 'mse':[]}
    
        for epoch in range(opt.n0_epoch, opt.N_epochs+1):
            eva_bch  = {'loss_G': 0, 'psnr': 0, 'ssim': 0, 'mse':0}
            generator.train()
            # discriminator.train()
            for i, dat in enumerate(data_train):
        
                # Configure model input
                dat_lr = Variable(dat["lr"].type(Tensor))
                dat_hr = Variable(dat["hr"].type(Tensor))
        
                # ------------------
                #  Train Generators
                # ------------------
                optimizer_G.zero_grad()
        
                # Generate a high resolution image from low resolution input
                gen_hr = generator(dat_lr)
        
                loss_content_pxl = my_loss(gen_hr,dat_hr,opt.nlm) # use pixel loss of data 

                # Total loss
                loss_G = loss_content_pxl 
        
                loss_G.backward()
                optimizer_G.step()
        
                # --------------
                #  Log Progress
                # --------------
                
                # loss for current batch before optimization 
                eva_bch['loss_G'] += loss_G.item() 
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                    % (epoch, opt.N_epochs, i, Nbatch, loss_G.item())
                )

            # evaluation using testing data. To save time directly use trainset
            # generator.eval()
            # with torch.no_grad():
            #     for i, dat in enumerate(data_test):
                    
            #         dat_lr = Variable(dat["lr"].type(Tensor))
            #         dat_hr = Variable(dat["hr"].type(Tensor))
            #         gen_hr = generator(dat_lr)
                    
                batch_mse = ((gen_hr - dat_hr) ** 2).data.mean()
                eva_bch['mse'] += batch_mse.item()
                batch_ssim = ssim_torch(gen_hr, dat_hr,data_range=1.0).item()
                eva_bch['ssim'] += batch_ssim
                # batch_psnr = 10 * log10((dat_hr.max()**2) / batch_mse)
                batch_psnr = 10 * log10(1.0 / batch_mse)
                eva_bch['psnr'] += batch_psnr
        
                if epoch % opt.sample_epoch == 0 and i % opt.sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    dat_lr = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor)
                    if ivar_lr == ivar_hr or nchl_i == nchl_o ==1: # same vars or 1 var to 1 var
                        img_grid = torch.cat((dat_lr, dat_hr,gen_hr), -1)
                    else:
                        img_grid = torch.cat((dat_hr,gen_hr), -1)
                    img_grid = img_grid.cpu()
                    img_grid = img_grid.detach().numpy()
    
                    nsubpfig = 6 # subfigure per figure
                    nfig = int(-(-len(img_grid) // nsubpfig))
                    for j in np.arange(nfig):
                        ne = min((j+1)*nsubpfig,len(img_grid))
                        ind = np.arange(j*nsubpfig,ne)
                        image = img_grid[ind,...]
                        ncol = 2
                        for k in range(nchl_o):
                            figname = out_path+"c%d_epoch%d_batch%d_id%d.png" % (ivar_hr[k]-3,epoch,i,j)
                            plt_sub(image,ncol,figname,k)
                
            # save model parameters
            if (opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and 
                epoch != opt.n0_epoch and epoch>=40): # do not save the first epoch, skip first 20 epochs
                torch.save({
                    'model_state_dict':generator.state_dict(),
                    'optimizer_state_dict':optimizer_G.state_dict()}, opath_nn+'netG_epoch_%d_re%d.pth' % (epoch,irep))

            # save loss\scores\psnr\ssim
            results['loss_G'].append(eva_bch['loss_G'] / Nbatch)
            results['psnr'].append(eva_bch['psnr'] / Nbatch) # use trainset instead of test, replace Nbatch_t
            results['ssim'].append(eva_bch['ssim'] / Nbatch) # 
            results['mse'].append(eva_bch['mse'] / Nbatch) # 
        
            # if epoch % opt.sample_interval == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'loss_G': results['loss_G'], 'psnr': results['psnr'], 'ssim': results['ssim'], 
                      'mse':results['mse']},
                index=range(opt.n0_epoch+1, epoch+2))
            data_frame.to_csv(opath_st + 'srf_%d_re%d' % (opt.up_factor,irep)+ '_train_results.csv', index_label='Epoch')

    


