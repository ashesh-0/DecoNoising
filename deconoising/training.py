import os
import socket

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pytorch_lightning.loggers import WandbLogger
from scipy.ndimage import gaussian_filter
from torch.nn import init

import deconoising.utils as utils
import wandb
from deconoising.learnable_gaussian_blur import GaussianLayer

############################################
#   Training the network
############################################


def getStratifiedCoords2D(numPix, shape):
    '''
    Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using startified sampling.
    '''
    box_size = np.round(np.sqrt(shape[0] * shape[1] / numPix)).astype(np.int32)
    coords = []
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    for i in range(box_count_y):
        for j in range(box_count_x):
            y = np.random.randint(0, box_size)
            x = np.random.randint(0, box_size)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                coords.append((y, x))
    return coords


def randomCropFRI(data, size, numPix, supervised=False, counter=None, augment=True):
    '''
    Crop a patch from the next image in the dataset.
    The patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    data: numpy array
        your dataset, should be a stack of 2D images, i.e. a 3D numpy array
    size: int
        witdth and height of the patch
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optinal): numpy array 
        This dataset could hold your target data e.g. clean images.
        If it is not provided the function will use the image from 'data' N2V style
    counter (optinal): int
        the index of the next image to be used. 
        If not set, a random image will be used.
    augment: bool
        should the patches be randomy flipped and rotated?
    
    Returns
    ----------
    imgOut: numpy array 
        Cropped patch from training data
    imgOutC: numpy array
        Cropped target patch. If dataClean was provided it is used as source.
        Otherwise its generated N2V style from the training set
    mask: numpy array
        An image holding marking which pixels should be used to calculate gradients (value 1) and which not (value 0)
    counter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''

    if counter is None:
        index = np.random.randint(0, data.shape[0])
    else:
        if counter >= data.shape[0]:
            counter = 0
            np.random.shuffle(data)

        index = counter
        counter += 1

    if supervised:
        img = data[index, ..., 0]
        imgClean = data[index, ..., 1]
        manipulate = False
    else:
        img = data[index]  #6X128X128
        imgClean = img
        manipulate = True

    imgOut, imgOutC, mask = randomCrop(img, size, numPix, imgClean=imgClean, augment=augment, manipulate=manipulate)

    return imgOut, imgOutC, mask, counter


def randomCrop(img, size, numPix, imgClean=None, augment=True, manipulate=True):
    '''
    Cuts out a random crop from an image.
    Manipulates pixels in the image (N2V style) and produces the corresponding mask of manipulated pixels.
    Patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    img: numpy array
        your dataset, should be a 2D image
    size: int
        witdth and height of the patch
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optinal): numpy array 
        This dataset could hold your target data e.g. clean images.
        If it is not provided the function will use the image from 'data' N2V style
    augment: bool
        should the patches be randomy flipped and rotated?
        
    Returns
    ----------    
    imgOut: numpy array 
        Cropped patch from training data with pixels manipulated N2V style.
    imgOutC: numpy array
        Cropped target patch. Pixels have not been manipulated.
    mask: numpy array
        An image marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    '''

    assert min(img.shape[-2:]) >= size
    # assert img.shape[1] >= size

    x = np.random.randint(0, img.shape[-1] - size)
    y = np.random.randint(0, img.shape[-2] - size)

    imgOut = img[..., y:y + size, x:x + size].copy()
    imgOutC = imgClean[..., y:y + size, x:x + size].copy()

    maxA = imgOut.shape[-1] - 1
    maxB = imgOut.shape[-2] - 1

    if manipulate:
        mask = np.zeros(imgOut.shape)
        hotPixels = getStratifiedCoords2D(numPix, imgOut.shape[-2:])
        for p in hotPixels:
            a, b = p[1], p[0]

            roiMinA = max(a - 2, 0)
            roiMaxA = min(a + 3, maxA)
            roiMinB = max(b - 2, 0)
            roiMaxB = min(b + 3, maxB)
            roi = imgOut[..., roiMinB:roiMaxB, roiMinA:roiMaxA]
            a_ = 2
            b_ = 2
            while a_ == 2 and b_ == 2:
                a_ = np.random.randint(0, roi.shape[-1])
                b_ = np.random.randint(0, roi.shape[-2])

            repl = roi[..., b_, a_]
            imgOut[..., b, a] = repl
            mask[..., b, a] = 1.0
    else:
        mask = np.ones(imgOut.shape)

    if augment:
        rot = np.random.randint(0, 4)
        imgOut = np.array(np.rot90(imgOut, rot, axes=(-2, -1)))
        imgOutC = np.array(np.rot90(imgOutC, rot, axes=(-2, -1)))
        mask = np.array(np.rot90(mask, rot, axes=(-2, -1)))
        if np.random.choice((True, False)):
            imgOut = np.array(np.flip(imgOut, axis=(-2, -1)))
            imgOutC = np.array(np.flip(imgOutC, axis=(-2, -1)))
            mask = np.array(np.flip(mask, axis=(-2, -1)))

    return imgOut, imgOutC, mask


def predict_for_one_img(img, net):
    inp = torch.Tensor(img[:, None]).cuda()
    psf_count = len(net)
    samples = []
    for psf_idx in range(psf_count):
        # Move normalization parameter to GPU
        stdTorch = torch.Tensor(np.array(net[psf_idx].std)).cuda()
        meanTorch = torch.Tensor(np.array(net[psf_idx].mean)).cuda()

        # Forward step
        output = net[psf_idx](
            (inp[psf_idx:psf_idx + 1] - meanTorch) / stdTorch) * 10.0  #We found that this factor can speed up training

        output = output * stdTorch + meanTorch
        samples.append(output[0].detach().cpu().numpy())
    return np.concatenate(samples, axis=0)


def trainingPred(my_train_data, net, dataCounter, size, bs, numPix, device, augment=True, supervised=True):
    '''
    This function will assemble a minibatch and process it using the a network.
    
    Parameters
    ----------
    my_train_data: numpy array
        Your training dataset, should be a stack of 2D images, i.e. a 3D numpy array
    net: a pytorch model
        the network we want to use
    dataCounter: int
        The index of the next image to be used. 
    size: int
        Witdth and height of the training patches that are to be used.
    bs: int 
        The batch size.
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    augment: bool
        should the patches be randomy flipped and rotated?
    Returns
    ----------
    samples: pytorch tensor
        The output of the network
    labels: pytorch tensor
        This is the tensor that was is used a target.
        It holds the raw unmanipulated patches.
    masks: pytorch tensor
        A tensor marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    dataCounter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''

    # Init Variables
    # inputs = torch.zeros(bs, 1, size, size)
    # labels = torch.zeros(bs, size, size)
    # masks = torch.zeros(bs, size, size)

    psf_count = my_train_data.shape[1]
    assert bs % psf_count == 0
    n_groups = bs // psf_count
    assert n_groups == 1, f'Set bs appropriately, ie, to {psf_count}'
    # Assemble mini batch
    im, l, m, dataCounter = randomCropFRI(my_train_data,
                                          size,
                                          numPix,
                                          counter=dataCounter,
                                          augment=augment,
                                          supervised=supervised)
    inputs = torch.Tensor(im[:, None])  #utils.imgToTensor(im)
    labels = torch.Tensor(l)  #utils.imgToTensor(l)
    masks = torch.Tensor(m)  #utils.imgToTensor(m)

    # Move to GPU
    inputs_raw, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
    samples_list = []
    for psf_idx in range(psf_count):
        # Move normalization parameter to GPU
        stdTorch = torch.Tensor(np.array(net[psf_idx].std)).to(device)
        meanTorch = torch.Tensor(np.array(net[psf_idx].mean)).to(device)

        # Forward step
        outputs = net[psf_idx]((inputs_raw[psf_idx:psf_idx + 1] - meanTorch) /
                               stdTorch) * 10.0  #We found that this factor can speed up training
        samples = (outputs).permute(1, 0, 2, 3)

        # Denormalize
        samples = samples * stdTorch + meanTorch
        samples_list.append(samples)

    samples = torch.cat(samples_list, dim=1)
    return samples, labels, masks, dataCounter


def apply_psf_list(samples, psf_list):
    psf_shape = psf_list[0].shape[2]
    pad_size = (psf_shape - 1) // 2
    assert psf_list[0].shape[2] == psf_list[0].shape[3]
    assert samples.shape[0] == 1, samples.shape
    rs = samples[0][:, None]
    # rs=torch.mean(samples,dim=0).reshape(samples.shape[1],1,samples.shape[2],samples.shape[3])
    # we pad the result
    # TODO: check the padding. I don't think one needs to pad in first two dimensions.
    rs = torch.nn.functional.pad(rs, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    # and convolve it with the psf
    conv_list = []
    for psf in psf_list:
        conv = torch.nn.functional.conv2d(rs, weight=psf, padding=0, stride=[1, 1])
        conv_list.append(conv)
    conv = torch.cat(conv_list, dim=1)
    return conv


def lossFunction(samples, labels, masks, std, psf_list, regularization, positivity_constraint, multipsf_w=0.005):
    assert samples.shape[0] == 1

    conv_outputs = []
    for i, psf in enumerate(psf_list):
        conv = apply_psf_list(samples[:, i:i + 1], [psf])
        conv_outputs.append(conv)

    conv = torch.cat(conv_outputs, dim=1)
    conv = conv[0]
    # This is the implementation of the positivity constraint
    signs = (samples < 0).float()
    samples_positivity_constraint = samples * signs
    pos_constraint_loss = positivity_constraint * torch.mean(torch.abs(samples_positivity_constraint)) / std
    # the N2V loss
    errors = (labels - conv)**2
    loss = torch.sum(errors * masks) / torch.sum(masks)
    n2v_loss = loss / (std**2)
    # TV regularization
    yprior = ((samples[:, :, 2:, 1:-1] - samples[:, :, :-2, 1:-1]) / 2.0)**2
    xprior = ((samples[:, :, 1:-1, 2:] - samples[:, :, 1:-1, :-2]) / 2.0)**2
    reg = torch.sqrt(yprior + xprior + 1e-15)  # total variation
    reg_loss = torch.mean(reg) * regularization / std
    # Similarity constraint
    idx = np.random.randint(samples.shape[1])
    # broadcasting is used here. so there is a warning, but that is alright.
    multipsf_loss = multipsf_w * torch.sqrt(1e-5 + torch.nn.MSELoss()(samples, samples[:, idx:idx + 1]))

    # print(f'N2V:{n2v_loss:.3f} Reg:{reg_loss/std:.3f} PosConst:{pos_constraint_loss/std:.3f} MultiPSF:{multipsf_loss:.3f}')
    net_loss = n2v_loss + reg_loss + pos_constraint_loss + multipsf_loss
    return {'net_loss': net_loss, 'multipsf_loss': multipsf_loss, 'n2v_loss': n2v_loss}


def artificial_psf(size_of_psf, std_gauss):
    filt = np.zeros((size_of_psf, size_of_psf))
    p = (size_of_psf - 1) // 2
    filt[p, p] = 1
    filt = torch.tensor(gaussian_filter(filt, std_gauss).reshape(1, 1, size_of_psf, size_of_psf).astype(np.float32))
    filt = filt / torch.sum(filt)
    return filt


def generate_psf_kernel_list(psf_count, relative_std, xy_squared_sum, net_0_std, kernel_size):
    psf_list = []
    for i in range(psf_count):
        factor = relative_std[i] / relative_std[0]
        kernel = GaussianLayer.generate_gaussian_kernel(xy_squared_sum, net_0_std * factor, kernel_size)
        psf_list.append(kernel)
    return psf_list


def trainNetwork(net,
                 trainData,
                 valData,
                 workdir,
                 device,
                 numOfEpochs=200,
                 stepsPerEpoch=50,
                 batchSize=4,
                 patchSize=100,
                 learningRate=0.0001,
                 lr_scheduler_patience=10,
                 numMaskedPixels=100 * 100 / 32.0,
                 virtualBatchSize=20,
                 valSize=20,
                 augment=True,
                 supervised=False,
                 psf_list=None,
                 psf_learnable=False,
                 psf_relative_std_list=None,
                 psf_kernel_size=None,
                 regularization=0.0,
                 positivity_constraint=1.0):
    '''
    Train a network using 
    
    
    Parameters
    ----------
    net: 
        The network we want to train.
        The number of output channels determines the number of samples that are predicted.
    trainData: numpy array
        Our training data. A 3D array that is interpreted as a stack of 2D images.
    valData: numpy array
        Our validation data. A 3D array that is interpreted as a stack of 2D images.
    postfix: string
        This identifier is attached to the names of the files that will be saved during training.
    device: 
        The device we are using, e.g. a GPU or CPU
    numOfEpochs: int
        Number of training epochs.
    stepsPerEpoch: int
        Number of gradient steps per epoch.
    batchSize: int
        The batch size, i.e. the number of patches processed simultainasly on the GPU.
    patchSize: int
        The width and height of the square training patches.
    learningRate: float
        The learning rate.
    numMaskedPixels: int
        The number of pixels that is to be manipulated/masked N2V style in every training patch.
    virtualBatchSize: int
        The number of batches that are processed before a gradient step is performed.
    valSize: int
        The number of validation patches processed after each epoch.
    augment: bool
        should the patches be randomy flipped and rotated? 
    supervised: bool
        Use this if you want to do supervised training instead of N2V
    psf_list: List[]
        This is the PSF that will be convolved with the predicted image during training
    regularization: float
        The weight for optional TV regularization of the deconvolved image.
    positivity_constraint: float
        The weight for the positivity constraint
        
    Returns
    ----------    
    trainHist: numpy array 
        A numpy array containing the avg. training loss of each epoch.
    valHist: numpy array
        A numpy array containing the avg. validation loss after each epoch.
    '''
    wandb.init()
    exptname = '/'.join(workdir.strip('/').split('/')[-3:])
    hostname = socket.gethostname()
    logger = WandbLogger(name=os.path.join(hostname, exptname), save_dir=workdir, project="Multi-PSF-Deconoising")
    if psf_learnable:
        # Create a list of learnable gaussian kernels.
        assert psf_list is None
        psf_count = len(psf_relative_std_list)
        rand_psf_std = 7.494  #np.random.rand() * 8
        kernel_size = psf_kernel_size
        net[0].gauss_layer = GaussianLayer(1, kernel_size=kernel_size, pad_type='reflect', std=rand_psf_std)
        xy_squared_sum = net[0].gauss_layer.xy_squared_sum
    else:
        psf_count = len(psf_list)

    # Calculate mean and std of data.
    # Everything that is processed by the net will be normalized and denormalized using these numbers.
    for i in range(psf_count):
        data_mean = np.mean(trainData[:, i])
        data_std = np.std(trainData[:, i])
        net[i].mean = data_mean
        net[i].std = data_std

    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     patience=lr_scheduler_patience,
                                                     factor=0.5,
                                                     verbose=True)

    running_loss = 0.0
    stepCounter = 0
    dataCounter = 0

    trainHist = []
    valHist = []

    while stepCounter / stepsPerEpoch < numOfEpochs:  # loop over the dataset multiple times
        losses = []
        n2v_losses = []
        multi_psf_losses = []
        stdev_outputs = []
        optimizer.zero_grad()
        stepCounter += 1

        # Loop over our virtual batch
        for a in range(virtualBatchSize):
            outputs, labels, masks, dataCounter = trainingPred(trainData,
                                                               net,
                                                               dataCounter,
                                                               patchSize,
                                                               batchSize,
                                                               numMaskedPixels,
                                                               device,
                                                               augment=augment,
                                                               supervised=supervised)
            avg_std = 0.0
            for single_net in net:
                avg_std += single_net.std / len(net)

            if psf_learnable:
                # weights have changed for the gaus_layer and so, kernel needs to be updated everytime.
                psf_list = generate_psf_kernel_list(psf_count, psf_relative_std_list, xy_squared_sum,
                                                    net[0].gauss_layer.std, psf_kernel_size)

            stdev_outputs += [outputs[:, i].std().item() for i in range(outputs.shape[1])]
            loss_dict = lossFunction(outputs, labels, masks, avg_std, psf_list, regularization, positivity_constraint)
            loss_dict['net_loss'].backward()
            running_loss += loss_dict['net_loss'].item()
            losses.append(loss_dict['net_loss'].item())
            n2v_losses.append(loss_dict['n2v_loss'].item())
            multi_psf_losses.append(loss_dict['multipsf_loss'].item())

        optimizer.step()

        # We have reached the end of an epoch
        if stepCounter % stepsPerEpoch == stepsPerEpoch - 1:
            running_loss = (np.mean(losses))
            losses = np.array(losses)
            logged_std = net[0].gauss_layer.std.item()
            utils.printNow("Epoch " + str(int(stepCounter / stepsPerEpoch)) +
                           f" finished. Current std: {logged_std:.3f}")
            utils.printNow(
                f"avg. loss: {np.mean(losses):.3f} +-(2SEM) {2.0 * np.std(losses) / np.sqrt(losses.size):.3f}",
                f'n2v:{np.mean(n2v_losses):.3f} multipsf:{np.mean(multi_psf_losses):.3f}, stdev:{np.mean(stdev_outputs):.2f}'
            )
            wandb.log({'std': logged_std})
            wandb.log({'loss': np.mean(losses)})
            wandb.log({'n2vloss': np.mean(n2v_losses)})
            wandb.log({'multipsf': np.mean(multi_psf_losses)})

            trainHist.append(np.mean(losses))
            losses = []
            fpath = os.path.join(workdir, "last_model.net")
            torch.save(net, fpath)

            valCounter = 0
            net.train(False)
            val_losses = []
            val_n2v_losses = []
            val_multipsf_losses = []

            for i in range(valSize):
                outputs, labels, masks, valCounter = trainingPred(valData,
                                                                  net,
                                                                  valCounter,
                                                                  patchSize,
                                                                  batchSize,
                                                                  numMaskedPixels,
                                                                  device,
                                                                  augment=augment,
                                                                  supervised=supervised)

                avg_std = 0.0
                for single_net in net:
                    avg_std += single_net.std / len(net)

                loss_dict = lossFunction(outputs, labels, masks, avg_std, psf_list, regularization,
                                         positivity_constraint)
                val_losses.append(loss_dict['net_loss'].item())
                val_n2v_losses.append(loss_dict['n2v_loss'].item())
                val_multipsf_losses.append(loss_dict['multipsf_loss'].item())

            net.train(True)
            avgValLoss = np.mean(val_losses)
            avgValn2vLoss = np.mean(val_n2v_losses)
            avgValMultiPsfLoss = np.mean(val_multipsf_losses)

            if len(valHist) == 0 or avgValLoss < np.min(np.array(valHist)):
                torch.save(net, os.path.join(workdir, f"best_model.net"))

            wandb.log({'ValLoss': avgValLoss, 'Valn2vLoss': avgValn2vLoss, 'ValMultiPsfLoss': avgValMultiPsfLoss})
            if ((1 + stepCounter) / stepsPerEpoch) % 20 == 0:
                log_imgs(valData, net)
            # convolved_img = None

            valHist.append(avgValLoss)
            scheduler.step(avgValLoss)
            epoch = (stepCounter / stepsPerEpoch)
            np.save(os.path.join(workdir, f"history_model.npy"), (np.array([np.arange(epoch), trainHist, valHist])))

    utils.printNow('Finished Training')
    return trainHist, valHist


def log_imgs(valData, net):
    inp = valData[0, ..., :64, :64]
    deconvolved_imgs = predict_for_one_img(inp, net)
    for i in range(len(deconvolved_imgs)):
        deco_image = wandb.Image(deconvolved_imgs[i, ..., None], caption="")

        wandb.log({f"DecoImgs_{i}": deco_image})
    inp_log = wandb.Image(inp[0, ..., None], caption='Input_0')
    wandb.log({'Input_0': inp_log})
