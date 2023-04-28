#!/usr/bin/env python3
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from unet.model import UNet
import torch
import deconoising.utils as utils
import deconoising.training as training
from tifffile import imread
from scipy.ndimage import gaussian_filter
import glob
import random
from deconoising.synthetic_data_generator import PSFspecify, create_dataset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="path to your training data and where network will be stored")
parser.add_argument("--fileName", help="name of your training data file", default='*.tif')
parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent)", default=30.0, type=float)
parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=100, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=10, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=1, type=int)
parser.add_argument("--virtualBatchSize", help="size of virtual batch", default=20, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=3, type=int)
parser.add_argument("--learningRate", help="initial learning rate", default=1e-3, type=float)
parser.add_argument("--netKernelSize", help="size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=64, type=int)
# parser.add_argument("--sizePSF", help="size of psf in pix, odd number", default=81, type=int)
parser.add_argument("--sizePSF", nargs='+', default=[])
# parser.add_argument("--stdPSF", help="size of std of gauss for psf", default=1.0, type=float)
parser.add_argument("--stdPSF", nargs='+', default=[])
parser.add_argument("--positivityConstraint", help="positivity constraint parameter", default=1.0, type=float)
parser.add_argument("--meanValue", help="mean value for the background ", default=0.0, type=float)

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)

# See if we can use a GPU
device = utils.getDevice()

print("args",str(args.name))

####################################################
#           PREPARE TRAINING DATA
####################################################
path = args.dataPath
# import pdb;pdb.set_trace()
fpath= os.path.join(args.dataPath, args.fileName)
data = []
assert fpath.split('.')[-1] == 'npz'
data_dict = np.load(fpath)
X_train = data_dict['X_train']
X_val = data_dict['X_val']


####################################################
#           PREPARE PSF
####################################################

def artificial_psf(size_of_psf, std_gauss):  
    filt = np.zeros((size_of_psf, size_of_psf))
    p = (size_of_psf - 1)//2
    filt[p,p] = 1
    filt = torch.tensor(gaussian_filter(filt,std_gauss).reshape(1,1,size_of_psf,size_of_psf).astype(np.float32))
    filt = filt/torch.sum(filt)
    return filt

sizePSF = [int(x) for x in args.sizePSF]
stdPSF = [float(x) for x in args.stdPSF]
psf_list = [PSFspecify(sizePSF[k],stdPSF[k]) for k in range(len(sizePSF))]

##################
# Augment the data
##################
# data = torch.Tensor(data[:,None])
my_train_data = create_dataset(torch.Tensor(X_train[:,None]), psf_list).numpy()
my_val_data = create_dataset(torch.Tensor(X_val[:,None]), psf_list).numpy()

####################################################
#           CREATE AND TRAIN NETWORK
####################################################
net = UNet(1, depth=args.netDepth)
# net.psf = psf_tensor.to(device)
# Split training and validation data

psf_tensor_list = [artificial_psf(psf.size, psf.std).to(device) for psf in psf_list]
# Start training
trainHist, valHist = training.trainNetwork(net = net, trainData = my_train_data, valData = my_val_data,
                                           postfix = args.name, directory = path,
                                           device = device, numOfEpochs = args.epochs, patchSize = args.patchSizeXY, stepsPerEpoch = 10,
					   virtualBatchSize = 20, batchSize = args.batchSize, learningRate = 1e-3,
                       psf_list = psf_tensor_list, 
					   positivity_constraint = args.positivityConstraint)
