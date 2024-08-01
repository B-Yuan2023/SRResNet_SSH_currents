# SRResNet_SSH_currents

This repository is for downscaling physical fields using 
super resolution residual neural network. 
Here the model is applied to downscale sea surface height (SSH) and 
depth-averaged currents in coastal region.

An example was constructed using minima dataset. 
Directories out2d_4 and out2d_16 contain samples of low and high resoltuion data. 

Batch files are used to run the model on the DKRZ Levante cluster.
The batch files contains the name of the parameter file (par*.py) to be read for runnning. 
All the parameter files and the corresponding batch fiels are in 'files_par_bash'.
To run the model, these files should be in the same directory as the main python 
scripts (srren.py for training, and test_epo.py for testing).
You can also modify the scripts such that parameter files can be read from a 
specified directory. 

To run the model without batch file, change line 23 'mod_name' from 'sys.argv[1]' 
to the name of the parameter file (e.g., 'par01') in srren.py and test_epo.py. 
Next under linux terminal: python srren.py, or run with python IDE like spyder. 

To train the model, python srren.py
To test the model, python test_epo.py

Inspired by 
https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan
and 
https://github.com/leftthomas/SRGAN

pytorch-msssim is from 
https://github.com/VainF/pytorch-msssim

Mainly used python scripts and steps to produce figures: 
Steps:
1. srren.py: run_ml_gpu1_r01_exp.sh
2. cal_metrics_intp.py: (not necessary to run before test_epo.py now)
	get 01,99,mean from direct interpolation
	and metrics, needed for test_epo.py, save all interpolated results
3. test_epo.py: run_test_epo_r01_exp.sh
	get metrics rmse/mae from each epoch (now comparison with direct interpolation removed)
   read_metrics.py: sort rmse and rmse_99 for all repeated runs. 
		get the index of epoch and run for optimal rmse. 
4. statistic_hr.py: 
	to obtain hour/batch of the maximum var (used in next step)
5. test_epo_user.py: (plot)
	plot 2d map of 01,99,mean for user epoch; plot snapshot of time series
	save 01,99,mean for user epoch (used in next step)
6. compare_metrics.py: (plot)
	read and compare global,mean,01,99 matrics
7. test_eop_user_st.py: (plot)
	plot distribution of all data (and at selected points)
8. test_user.py: (plot) 
	plot 2D map with mae&rmse at selected times
	plot time evolution of mae&rmse along batches 
