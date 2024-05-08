# -*- coding: utf-8 -*-


import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import util
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader

from skimage.transform import swirl, warp
import copy

from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = self.__image_list()


    def __image_list(self):
        image_list = []
        for a,b,c in os.walk(self.data_dir):
            if len(c)>0:
                for i in c:
                    if i.endswith(('jpg', 'webp', 'png', 'jpeg')):
                        image_list.append(os.path.join(a,i))
        
        return image_list

    def __len__(self):
        return len(self.image_list)
    

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name)
        image = resize(image)
        if self.transform:
            image = self.transform(image)
        return image, self.image_list[idx]

def resize(image, min_size = 512):
    w,h = image.size
    new_shape = (min_size, int(min_size*h/w)) if w < h else (int(min_size*w/h), min_size) 
    return image.resize(new_shape)

def cv2_read_image(image_path):
    image = cv2.imread(image_path)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    chs = cv2.split(image)
    return cv2.merge((chs[0], chs[1], chs[2]))

def get_image_paths(based_dir):
    
    image_paths = []
    
    for dirpath, dirnames, filenames in os.walk(based_dir):
        if len(filenames)>0:
            imgs = [os.path.join(dirpath, f) for f in filenames \
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.webp')) ]
            
            image_paths.extend(imgs)
        
    return image_paths


def save_image(filename, image):
    cv2.imwrite(filename, image)
    

def show_img(image:torch.Tensor, cmap='gray'):
    if type(image) is torch.Tensor:
        plt.imshow(image.permute(1, 2, 0), cmap='gray')
        return
    
    try:
        plt.imshow(image, cmap='gray')
    except:
        return


def img2liquify(image):
    """

    Parameters
    ----------
    image : Should be Pillow format
        DESCRIPTION.

    Returns
    -------
    liquified_image : TYPE
        DESCRIPTION.

    """
    w, h = image.size
    image  = np.asarray(image)
    liquified_image = copy.deepcopy(image)
    for i in range(np.random.randint(10,20)):
       x = np.random.randint(0,w)
       y = np.random.randint(0,h)
       strength = np.random.randint(4,10)
       radius = np.random.randint(int(min(w, h)/20),int(min(w,h)/5))
       liquified_image = swirl(image=liquified_image, rotation=0, strength=strength, radius=radius, center=(x,y))
       liquified_image = (liquified_image - liquified_image.min()) * (255 / (liquified_image.max() - liquified_image.min()))
       liquified_image = liquified_image.astype(np.uint8)

    return  Image.fromarray(liquified_image)




# =============================================================================
# 
# # Đọc ảnh đầu vào
# image = Image.open('D:/K32/do_an_tot_nghiep/AIGCDetectBenchmark/images/dog.jpg')
# image_liquify = img2liquify(image)
# plt.imshow(image)
# 
# plt.imshow(image_liquify)
# # Thiết lập các tham số cho hiệu ứng liquify
# strength = 1  # Độ mạnh của hiệu ứng
# radius = 100    # Bán kính của vùng bóp méo
# 
# # Áp dụng hiệu ứng liquify bằng hàm swirl từ scikit-image
# w, h = image.size
# image  = np.asarray(image)
# 
# liquified_image = copy.deepcopy(image)
# for i in range(np.random.randint(10,15)):
#    x = np.random.randint(0,w)
#    y = np.random.randint(0,h)
#    strength = np.random.randint(1,3)
#    radius = np.random.randint(30,100)
#    liquified_image = swirl(liquified_image, rotation=0, strength=strength, radius=radius, center=(x,y))
# 
# =============================================================================


if __name__ == '__main__':
    
    
    
    
    parser = argparse.ArgumentParser(description='Local image processing')
    parser.add_argument('--input_dir', default='D:/K32/do_an_tot_nghiep/data/real_gen_dataset',
        type=str, help='Directory of images need to convert')
    parser.add_argument('--result_dir', default='D:/K32/do_an_tot_nghiep/data/real_gen_dataset_liquify',
        type=str, help='Directory for results')
    
    args = parser.parse_args()
    
    #util.mkdir(args.result_dir)
    

    dataset = CustomImageDataset(args.input_dir, transform=img2liquify)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    len(dataset)

    for i, (img, name) in enumerate(dataset):
        file_path = name.replace(args.input_dir, args.result_dir)
        img.save(file_path)
        
    
# =============================================================================
#     for path in tqdm(image_paths[6:10]):
#         img = Image.open(path)
#         w, h = img.size
#         new_w = (w<=h) * 512 + (w>h) * 512 * w/h
#         new_h = (w>=h) * 512 + (w<h) * 512 * w/h
#         img = img.resize((int(new_w) , int(new_h)))
# 
#         rel_path = os.path.relpath(path, args.input_dir)
#         
#         dst_path = os.path.join(args.result_dir, rel_path)
#         
#         util.mkdir(os.path.dirname(dst_path))
#         
#         image = img2liquify(img)
#         
#         image.save(dst_path)
#         #save_image(dst_path, image)
# =============================================================================
        
        
        
        
        
        
        
    