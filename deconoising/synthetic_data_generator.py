import numpy as np
import torch
from scipy.ndimage import gaussian_filter

def artificial_psf(size_of_psf , std_gauss):  
    filt = np.zeros((size_of_psf, size_of_psf))
    p = (size_of_psf - 1)//2
    filt[p,p] = 1
    filt = torch.tensor(gaussian_filter(filt,std_gauss).reshape(1,1,size_of_psf,size_of_psf).astype(np.float32))
    filt = filt/torch.sum(filt)
    return filt

def convolve_with_psf(data, psf):
    psf_shape = psf.shape[2]
    pad_size = (psf_shape - 1)//2
    assert psf.shape[2] == psf.shape[3]
    # rs=torch.mean(data,dim=0).reshape(data.shape[1],1,data.shape[2],data.shape[3])
    # we pad the result
    # rs should be (N,1,H,W)
    rs=torch.nn.functional.pad(data,(pad_size, pad_size, pad_size, pad_size),mode='reflect')
    # and convolve it with the psf
    conv=torch.nn.functional.conv2d(rs,
                                    weight=psf,
                                    padding=0,
                                    stride=[1,1])
    return conv

class PSFspecify:
    def __init__(self, size, std) -> None:
        self.size = size
        self.std = std

def create_dataset(images, psf_specification_list, pixel_independent_gaussian_noise_std = None):
    diff_psf_outputs = []
    for psf_specification in psf_specification_list:
        psf = artificial_psf(psf_specification.size, psf_specification.std)
        output_images = convolve_with_psf(images,psf)
        if pixel_independent_gaussian_noise_std:
            output_images=output_images + (pixel_independent_gaussian_noise_std**0.5)*torch.randn(*output_images.shape)
        
        assert output_images.shape[1] ==1
        diff_psf_outputs.append(output_images)
    
    multi_psf_output = torch.cat(diff_psf_outputs,dim=1)
    return multi_psf_output


