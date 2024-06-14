import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from random import random, choice
import copy
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from tqdm import tqdm

#from data.process import FourierTransform

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073],
    #Mean: tensor([0.4961, 0.4607, 0.4350])
    #Std: tensor([0.2461, 0.2345, 0.2351])
    #"imagenet":[0.4961, 0.4607, 0.4350],
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711],
    #"imagenet":[0.2461, 0.2345, 0.2351],
}


class FourierTransform(nn.Module):
    def __init__(self, cutoff=[0.0, 1.0]):
        super(FourierTransform, self).__init__()
        self.cutoff = cutoff
        self.shape = None
        self.mask = None

    def forward(self, img):

        if self.shape != img.shape:
            self.shape =  img.shape
            self.mask = self.create_filter(self.shape, self.cutoff)
    
        # Apply Fourier Transform
        img_fft = torch.fft.fft2(img)
        img_fft_shifted = torch.fft.fftshift(img_fft)
        
        # Compute Magnitude and Phase
        magnitude = torch.abs(img_fft_shifted)
        phase = torch.angle(img_fft_shifted)
        
        #magnitude[:] = magnitude.mean()
         
        # Combine Magnitude and Phase back to the complex form
        img_fft_shifted = torch.polar(magnitude, phase)
        
        # Optionally apply a low-pass filter

        img_fft_shifted = img_fft_shifted * self.mask
        
        # Apply Inverse Fourier Transform
        img_fft_shifted = torch.fft.ifftshift(img_fft_shifted)
        img_ifft = torch.fft.ifft2(img_fft_shifted)
        
        # Take the real part of the inverse FFT result
        img_processed = torch.real(img_ifft)
        return img_processed
    
    def create_filter(self, shape, cutoff=[0.0, 1.0]):
        # Create a low-pass filter mask
        rows, cols = shape[-2], shape[-1]
        crow, ccol = rows // 2 , cols // 2  # center
        
        # Create a circular low-pass filter mask
        y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')

        
        r1 = int(cutoff[0] * min(crow, ccol))
        r2 = int(cutoff[1] * min(crow, ccol))
        assert r1 < r2, "r1 must be less than r2"
        
        mask1 = ((y - crow) ** 2 + (x - ccol) ** 2) >= r1 ** 2
        mask2 = ((y - crow) ** 2 + (x - ccol) ** 2) <= r2 ** 2
        mask = mask1 & mask2

        mask = mask.float()
        
        return mask
        
###############################################################################



class ConvertToL:
    def __call__(self, img):
        return img.convert('L')
    

def custom_resize(img):
    if min(img.size) < 256:
        img = transforms.Resize(256)(img) 
        return img
    else:
        return img
    
fouries_fz = FourierTransform(cutoff=[0.0, 0.75])

rz_func = transforms.Lambda(lambda img: custom_resize(img))

trans = transforms.Compose([
            rz_func,
            transforms.CenterCrop(224),
            #ConvertToL(),
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet']),  # Normalize to [-1, 1]
            fouries_fz,
            #transforms.Normalize(mean=0.5, std=0.5),

            #transforms.CenterCrop(224),
            #transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),

            ])

root = r'D:/K32\do_an_tot_nghiep/data/real_gen_dataset/train'


dataset = datasets.ImageFolder(root=root, transform=trans)

# =============================================================================
# real_ims = []
# fake_ims = []
# for im, label in tqdm(dataset):
#     if label == 0:
#         real_ims.append(im)
#     else:
#         fake_ims.append(im)
# 
# =============================================================================


im, a = dataset[np.random.choice(range(len(dataset)))]
im = im.permute((1,2,0))

plt.imshow(im)
plt.title(a)

im.max()
np.sum(np.abs(np.asarray(im)))

plt.imshow(np.log(im + 1), cmap='gray')  # Apply logarithmic scaling for better visualization

# =============================================================================
# 
# mean_image_tensor = np.mean(real_ims, axis=0)
# mean_image_tensor = np.transpose(mean_image_tensor, axes=(1,2,0))
# 
# mean_fake_ims = np.mean(fake_ims, axis=0)
# mean_fake_ims = np.transpose(mean_fake_ims, axes=(1,2,0))
# 
# plt.imshow(np.log(mean_image_tensor+1), cmap='gray')
# plt.imshow(np.log(mean_fake_ims+1), cmap='gray')
# 
# plt.imshow(mean_image_tensor, cmap='gray')
# plt.imshow(mean_fake_ims, cmap='gray')
# 
# 
# =============================================================================

