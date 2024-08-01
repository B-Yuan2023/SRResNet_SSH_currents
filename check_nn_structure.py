#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:16:52 2024
check structure of the model 
@author: g260218
"""

import os
import torch
from models import GeneratorResNet


if __name__ == '__main__':

    nchl_i,nchl_o = 1,1
    nchl = nchl_o
    
    residual_blocks=6
    up_factor=16
    hr_shape = (128, 128)
    lr_shape = (int(hr_shape[0]/up_factor), int(hr_shape[1]/up_factor))

    # Initialize generator and discriminator
    generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                n_residual_blocks=residual_blocks,up_factor=up_factor)
    
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Total number of learnable parameters: {num_params}")
    # print(generator) # model infomation of each layer, without no. parameters. 
    
    batch_size = 12 
    dat_lr = torch.randn(batch_size, nchl_i, lr_shape[0], lr_shape[1]) 
    gen_hr = generator(dat_lr)
    
    # from torchsummary import summary
    # summary(generator, dat_lr.shape)
    
    # from torchviz import make_dot
    # Save the graph to a file (optional)
    # dot = make_dot(gen_hr, params=dict(generator.named_parameters()))
    # out_path = 'nn_structure/'
    # os.makedirs(out_path, exist_ok=True)
    # ofname = "nn_b%d_s%d_batch%d_nch%d_lr%d_%d" % (
    #     residual_blocks,up_factor,batch_size,nchl_i,lr_shape[0],lr_shape[1]) + '.png'
    # dot.render(out_path+ofname, format='png')

    # # Display the graph
    # dot.view()
    
    # from torchview import draw_graph
    # model_graph = draw_graph(generator(), input_size=(1,3,224,224), expand_nested=True)
    # model_graph.visual_graph
    