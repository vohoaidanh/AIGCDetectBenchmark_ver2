import cv2
import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile

import torchvision
import os
import copy
import torch
from scipy import fftpack
import imageio
from skimage.transform import resize
from .process import *
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize) # 随机剪裁，默认224
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img) # 不处理
    else:
        crop_func = transforms.CenterCrop(opt.cropSize) # 中心裁剪

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path

##############################################################################
class read_data_custom():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]
        
        #Image 2
        idx2 = index
        while(idx2 == index):
            idx2 = choice(range(len(self.real_img_list)))
        
        img2 = Image.open(self.img[idx2]).convert('RGB')
        # compute scaling
        
        height, width = img.height, img.width
        if (not self.opt.isTrain) and (not self.opt.isVal):
            img = custom_augment(img, self.opt)
            img2 = custom_augment(img2, self.opt)

        img = processing(img,self.opt,'imagenet')
        img2 = processing(img2,self.opt,'imagenet')
        
        return img, img2, target

    def __len__(self):
        return len(self.label)
    
class read_data_combine():
    """
    root1, root2: should be a full_path with slip set like
    D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train
    Note! if root1 != root2, the dataset filename coressponing should be the same
    
    """
    
    def __init__(self, opt):
        opt1 = copy.deepcopy(opt)
        opt1.detect_method = opt.method_combine.split('+')[0]
        #get the current split [train, val, test]
        split = os.path.basename(opt.dataroot.rstrip('/'))

        opt2 = copy.deepcopy(opt)
        opt2.dataroot = '{}/{}/'.format(opt.dataroot2, split)
        opt2.detect_method = opt.method_combine.split('+')[-1]
        
        if opt2.detect_method == 'FreDect':
            opt2.dct_mean = torch.load('./weights/auxiliary/dct_mean').permute(1,2,0).numpy()
            opt2.dct_var = torch.load('./weights/auxiliary/dct_var').permute(1,2,0).numpy()
        
        self.dataset1 = read_data(opt=opt1)
        self.dataset2 = read_data(opt=opt2)
           
        # Ensure that the number of samples in both datasets are the same
        assert len(self.dataset1) == len(self.dataset2), \
            f"Number of samples in both datasets must be the same.{len(self.dataset1)} != {len(self.dataset2)}"

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        # Load samples from both datasets at the same index
        img1, label = self.dataset1.__getitem__(idx)
        img2, _ = self.dataset2.__getitem__(idx)

        # Return images and labels
        return img1, img2, label

class read_data_cnnspot_fredect():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        img2 = copy.deepcopy(img)
        imgname = self.img[index]
        # compute scaling
        height, width = img.height, img.width
        if (not self.opt.isTrain) and (not self.opt.isVal):
            img = custom_augment(img, self.opt)
            img2 = custom_augment(img2, self.opt)

        
        
        img = processing(img,self.opt,'imagenet')
        img2 = processing_DCT(img2,self.opt)
        

        return img, img2, target

    def __len__(self):
        return len(self.label)    


class shading_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt, rgb_dir='rgb', shading_dir = 'shading' ):
        """
        Parameters
        ----------
        opt : TYPE
            DESCRIPTION.
        split : [train, test, val]
            DESCRIPTION. The default is 'train'.
        rgb_dir : dir of RGB images
            DESCRIPTION. The default is 'rgb'.
        shading_dir : dir of shading images
            DESCRIPTION. The default is 'shading'.

        Returns Dataset
        -------

        """
        self.opt = opt
        self.root = os.path.dirname(opt.dataroot.rstrip('/'))
        self.rgb_dir = rgb_dir
        self.shading_dir = shading_dir
        self.split = os.path.basename(opt.dataroot.rstrip('/'))
        
        real_rgb_name = os.listdir(os.path.join(self.root, self.rgb_dir, self.split, '0_real'))
        real_label_list = [0 for _ in range(len(real_rgb_name))]
        
        real_rgb_list = [os.path.join(self.root, self.rgb_dir, self.split, '0_real',i) \
                         for i in real_rgb_name if i.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
        fake_rgb_name = os.listdir(os.path.join(self.root, self.rgb_dir, self.split, '1_fake'))
        fake_rgb_list = [os.path.join(self.root, self.rgb_dir, self.split, '1_fake',i) \
                         for i in fake_rgb_name if i.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]        
        
        
        fake_label_list = [1 for _ in range(len(fake_rgb_name))]
                    
        self.input = real_rgb_list + fake_rgb_list
        self.shading = [i.replace(self.rgb_dir, self.shading_dir) for i in self.input]
        self.labels = real_label_list + fake_label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                    
        rgb  = Image.open(self.input[idx]).convert('RGB')
        shading = Image.open(self.shading[idx]).convert('RGB')
        
        target  = self.labels[idx]
        
        if (not self.opt.isTrain) and (not self.opt.isVal):
            rgb = custom_augment(rgb, self.opt)
            shading = custom_augment(shading, self.opt)
            
        #if self.opt.detect_method.lower() in ['shading']:
        rgb = processing(rgb,self.opt,'imagenet')
        shading = processing(shading,self.opt,'imagenet')
            
        #else:
            #raise ValueError(f"Unsupported model_type: {self.opt.detect_method}")
        
        return rgb, shading, target
    
class midas_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt, rgb_dir='rgb', shading_dir = 'shading' ):
        """
        Parameters
        ----------
        opt : TYPE
            DESCRIPTION.
        split : [train, test, val]
            DESCRIPTION. The default is 'train'.
        rgb_dir : dir of RGB images
            DESCRIPTION. The default is 'rgb'.
        shading_dir : dir of shading images
            DESCRIPTION. The default is 'shading'.

        Returns Dataset
        -------

        """
        self.opt = opt
        self.root = os.path.dirname(opt.dataroot.rstrip('/'))
        self.rgb_dir = rgb_dir
        self.shading_dir = shading_dir
        self.split = os.path.basename(opt.dataroot.rstrip('/'))
        
        real_rgb_name = os.listdir(os.path.join(self.root, self.rgb_dir, self.split, '0_real'))
        real_label_list = [0 for _ in range(len(real_rgb_name))]
        
        real_rgb_list = [os.path.join(self.root, self.rgb_dir, self.split, '0_real',i) \
                         for i in real_rgb_name if i.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
        fake_rgb_name = os.listdir(os.path.join(self.root, self.rgb_dir, self.split, '1_fake'))
        fake_rgb_list = [os.path.join(self.root, self.rgb_dir, self.split, '1_fake',i) \
                         for i in fake_rgb_name if i.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]        
        
        
        fake_label_list = [1 for _ in range(len(fake_rgb_name))]
                    
        self.input = real_rgb_list + fake_rgb_list
        self.shading = [i.replace(self.rgb_dir, self.shading_dir) for i in self.input]
        self.labels = real_label_list + fake_label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # In this method, shading is MiDas image
        rgb  = Image.open(self.input[idx]).convert('RGB')
        shading = Image.open(self.shading[idx]).convert('RGB')
        
        mask = create_mask(rgb, shading, thres = 128, background = True) # Get background only
        
        target  = self.labels[idx]
        
        if (not self.opt.isTrain) and (not self.opt.isVal):
            mask = custom_augment(mask, self.opt)
            
        #if self.opt.detect_method.lower() in ['shading']:
        mask = processing(mask,self.opt,'imagenet')
        
            
        #else:
            #raise ValueError(f"Unsupported model_type: {self.opt.detect_method}")
        
        return mask, target
    
##############################################################################


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    img = torchvision.transforms.Resize((opt.loadSize,opt.loadSize))(img) 
    return img


def custom_augment(img, opt):
    
    # print('height, width:'+str(height)+str(width))
    # resize
    if opt.noise_type=='resize':
        
        height, width = img.height, img.width
        img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 

    img = np.array(img)
    # img = img[0:-1:4,0:-1:4,:]
    if opt.noise_type=='blur':
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if opt.noise_type=='jpg':
        
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    
    return Image.fromarray(img)


def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths

def process_img(img,opt,imgname,target):
    if opt.detect_method in ['CNNSpot','Gram']:
        img = processing(img,opt,'imagenet')
    elif opt.detect_method == 'FreDect':
        img = processing_DCT(img,opt)
    elif opt.detect_method == 'Fusing':
        input_img, cropped_img, scale = processing_PSM(img,opt)
        return input_img, cropped_img, target, scale, imgname
    elif opt.detect_method == 'LGrad':
        opt.cropSize=256
        img = processing_LGrad(img, opt.gen_model, opt)
    elif opt.detect_method == 'LNP':
        img = processing_LNP(img, opt.model_restoration, opt, imgname)
    elif opt.detect_method == 'DIRE':
        img = processing_DIRE(img,opt,imgname)
    elif opt.detect_method == 'UnivFD':
            img = processing(img, opt,'clip')
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")


    return img, target


class read_data_cam():
    """
        This dataset for CNNSpot_CAM detect method
        Note! Positive class 1 = Real image
    """
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        real_label_list = [1 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [0 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list
        self.real_img_list = real_img_list
        self.length = self.__len__()
        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        
        img, label = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]
        # compute scaling
        height, width = img.height, img.width
        
        if (not self.opt.isTrain) and (not self.opt.isVal):
            # In eval mode
            img = custom_augment(img, self.opt)
            img = processing(img,self.opt,'imagenet')
            
            if self.opt.CNNSpot_CAM_inference:
                # Being in inference mode means we need the model's response image to be real or fake
                return img, img, label
            else:
                if random() > 0.5:      
                    idx2 = choice(range(len(self.label)))
                else:
                    idx2 = index
                    
                if idx2!=index:
                    img2, label2 = Image.open(self.img[idx2]).convert('RGB'), self.label[idx2]
                    img2 = custom_augment(img2, self.opt)
                    img2 = processing(img2,self.opt,'imagenet')
                    return img, img2, 0
                else:
                    return img, img, 1

      
        img = processing(img,self.opt,'imagenet')
            
        if random() > 0.5:      
            idx2 = choice(range(len(self.label)))
        else:
            idx2 = index
        
        if idx2!=index:
            img2, label2 = Image.open(self.img[idx2]).convert('RGB'), self.label[idx2]
            img2 = processing(img2,self.opt,'imagenet')
            return img, img2, 0
            
        else:
            img2 = img
            return img, img2, 1
        
        return None

    def __len__(self):
        if (not self.opt.isTrain and not self.opt.isVal):
            return len(self.label)
        else:
            return len(self.real_img_list)
    
    
class read_data():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))

    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]
        
        # compute scaling
        height, width = img.height, img.width
        if (not self.opt.isTrain) and (not self.opt.isVal):
            img = custom_augment(img, self.opt)
      
      
        if self.opt.detect_method in ['CNNSpot','Gram','Steg', 'CNNSpot_CAM', 'Resnet_Attention']:
            img = processing(img,self.opt,'imagenet')
        elif self.opt.detect_method == 'FreDect':
            img = processing_DCT(img,self.opt)
        elif self.opt.detect_method == 'Fusing':
            input_img, cropped_img, scale = processing_PSM(img,self.opt)
            return input_img, cropped_img, target, scale, imgname
        elif self.opt.detect_method == 'LGrad':
            self.opt.cropSize=256
            img = processing_LGrad(img,self.opt.gen_model,self.opt)
        elif self.opt.detect_method == 'LNP':
            img = processing_LNP(img,self.opt.model_restoration,self.opt,imgname)
        elif self.opt.detect_method == 'DIRE':
            img = processing_DIRE(img,self.opt,imgname)
        elif self.opt.detect_method == 'UnivFD':
            img = processing(img,self.opt,'clip')
        elif self.opt.detect_method == 'Derivative':
            img = processing_DER(img,self.opt,'imagenet')
        elif self.opt.detect_method == "CNNSpot_Noise":
            if random() < 0.3 and target == 1 and (self.opt.isTrain or self.opt.isVal):
                noise_data = np.random.randint(0, 256, size=(256,256,3), dtype=np.uint8)  # Tạo dữ liệu nhiễu ngẫu nhiên từ 0 đến 255
                img = Image.fromarray(noise_data) 
            img = processing(img,self.opt,'imagenet')
        
        elif self.opt.detect_method in ["CNNSimpest"]:
            img = processing_CNNSimpest(img,self.opt,'imagenet')
            
        elif self.opt.detect_method in ["Resnet_Metric"]:
            #img = processing_Resnet_Metric(img, self.opt, 'imagenet')
            img = processing_Resnet_Metric_DCT(img, self.opt, 'imagenet')
            
        else:
            raise ValueError(f"Unsupported model_type: {self.opt.detect_method}")


        return img, target

    def __len__(self):
        return len(self.label)
    

def read_data_new(opt):
    if opt.method_combine is not None:
        if 'shading' in opt.method_combine.lower():
            return shading_dataset(opt)
        if 'fredect' in opt.method_combine.lower():
            opt.dct_mean = torch.load('./weights/auxiliary/dct_mean').permute(1,2,0).numpy()
            opt.dct_var = torch.load('./weights/auxiliary/dct_var').permute(1,2,0).numpy()
            return read_data_cnnspot_fredect(opt)
        if 'midas' in opt.method_combine.lower():
            return midas_dataset(opt)
        
    if 'cnnspot_cam' in opt.detect_method.lower():
        return read_data_cam(opt)
    else:
        return read_data(opt)
    
