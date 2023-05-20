#!/usr/bin/env python3
"""
Example run: 
python script_training.py --dataPath=/home/ubuntu/ashesh/data/Flywing/Flywing_n10/train/ --fileName=train_data.npz 
--rootWorkDir=/home/ubuntu/ashesh/training/deconoising/ --batchSize=4 --sizePSF 81 81 81 81 --stdPSF 3 3 3 3 --epochs=1000 --lr_scheduler_patience=50
"""
import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tifffile import imread

import deconoising.training as training
import deconoising.utils as utils
from deconoising.synthetic_data_generator import PSFspecify, create_dataset
from deconoising.training import artificial_psf
from deconoising.workdir_manager import add_git_info, get_workdir
from unet.model import UNet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="path to your training data and where network will be stored")
parser.add_argument("--rootWorkDir", help="directory where network will be stored")
parser.add_argument("--fileName", help="name of your training data file", default='*.tif')
parser.add_argument("--validationFraction",
                    help="Fraction of data you want to use for validation (percent)",
                    default=30.0,
                    type=float)
parser.add_argument("--learnable_psf",
                    default=True,
                    type=bool,
                    help='This is used to create a learnable gaussian with which one deconvolves')

parser.add_argument("--learnable_psf_init_std",
                    default=None,
                    type=float,
                    help='This is the value of the stdev() of the psf you initialize with.')

parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=100, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=10, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=1, type=int)
parser.add_argument("--virtualBatchSize", help="size of virtual batch", default=20, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=3, type=int)
parser.add_argument("--learningRate", help="initial learning rate", default=1e-3, type=float)
parser.add_argument("--lr_scheduler_patience", help="learning rate scheduler patience", default=50, type=int)
parser.add_argument("--netKernelSize", help="size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=64, type=int)
# parser.add_argument("--sizePSF", help="size of psf in pix, odd number", default=81, type=int)
parser.add_argument("--sizePSF", nargs='+', default=[])
parser.add_argument("--noiseStd", help="stdev of noise which is added to input", default=None, type=float)
parser.add_argument("--multipsf_loss_w", help="stdev of noise which is added to input", default=0.01, type=float)

# parser.add_argument("--stdPSF", help="size of std of gauss for psf", default=1.0, type=float)
parser.add_argument("--stdPSF", nargs='+', default=[])
parser.add_argument("--positivityConstraint", help="positivity constraint parameter", default=1.0, type=float)
parser.add_argument("--meanValue", help="mean value for the background ", default=0.0, type=float)
parser.add_argument("--use_max_version", default=False, help="Overwrite the max version of the model")

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)

# See if we can use a GPU
device = utils.getDevice()

print("args", str(args.name))

sizePSF = [int(x) for x in args.sizePSF]
stdPSF = [float(x) for x in args.stdPSF]
psf_list = [PSFspecify(sizePSF[k], stdPSF[k]) for k in range(len(sizePSF))]
psf_tensor_list = None if args.learnable_psf else [artificial_psf(psf.size, psf.std).to(device) for psf in psf_list]
psf_std = [psf.std for psf in psf_list]
workdir = get_workdir({'name': args.name, 'psf_list': psf_std}, args.rootWorkDir, args.use_max_version)
pixel_independent_gaussian_noise_std = args.noiseStd
print('')
print('Saving model & config to', workdir)
print('')
####################################################
#           PREPARE TRAINING DATA
####################################################
# path = args.dataPath
# import pdb;pdb.set_trace()
fpath = os.path.join(args.dataPath, args.fileName)
data = []
assert fpath.split('.')[-1] == 'npz'
data_dict = np.load(fpath)
X_train = data_dict['X_train']
X_val = data_dict['X_val']

####################################################
#           PREPARE PSF
####################################################

##################
# Augment the data
##################
# data = torch.Tensor(data[:,None])
my_train_data = create_dataset(torch.Tensor(X_train[:, None]),
                               psf_list,
                               pixel_independent_gaussian_noise_std=pixel_independent_gaussian_noise_std).numpy()
my_val_data = create_dataset(torch.Tensor(X_val[:, None]), psf_list).numpy()

####################################################
#           CREATE AND TRAIN NETWORK
####################################################
nets = nn.ModuleList([UNet(1, depth=args.netDepth) for _ in range(len(psf_list))])

with open(os.path.join(workdir, 'config.json'), 'w') as fp:
    config = {'psf': [(psf.size, psf.std) for psf in psf_list]}
    add_git_info(config)
    json.dump(config, fp)

trainHist, valHist = training.trainNetwork(net=nets,
                                           trainData=my_train_data,
                                           valData=my_val_data,
                                           workdir=workdir,
                                           device=device,
                                           numOfEpochs=args.epochs,
                                           patchSize=args.patchSizeXY,
                                           stepsPerEpoch=10,
                                           virtualBatchSize=20,
                                           batchSize=args.batchSize,
                                           learningRate=1e-3,
                                           lr_scheduler_patience=args.lr_scheduler_patience,
                                           psf_list=psf_tensor_list,
                                           psf_learnable=args.learnable_psf,
                                           psf_learnable_init_std=args.learnable_psf_init_std,
                                           psf_relative_std_list=[psf.std / psf_list[0].std for psf in psf_list],
                                           psf_kernel_size=psf_list[0].size,
                                           multipsf_loss_w=args.multipsf_loss_w,
                                           positivity_constraint=args.positivityConstraint)
