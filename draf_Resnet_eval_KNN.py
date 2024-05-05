# -*- coding: utf-8 -*-
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import euclidean_distances
from sklearn.metrics import confusion_matrix


from networks.resnet import resnet50
import torch
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image

from options import TrainOptions
from data.datasets import read_data_combine
from data import create_dataloader_new,create_dataloader
import numpy as np

import matplotlib.pyplot as plt


model = resnet50(num_classes=1)
state_dict = torch.load(r"D:\K32\do_an_tot_nghiep\data\240428_CNNSpot_checkpoint\model_epoch_best.pth", map_location='cpu')
model.load_state_dict(state_dict['model'])
features = torch.nn.Sequential(*list(model.children())[:8])

opt = TrainOptions().parse()
#opt.dataroot = r'D:\K32\do_an_tot_nghiep\data\RealFakeDB512s'
opt.dataroot = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot2 = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.val_split)
opt.batch_size = 1
opt.method_combine = None
opt.isTrain = False
opt.isVal = False
opt.noise_type = 'None'
opt.num_threads = 0

data_loader = create_dataloader_new(opt)
features.eval()

def average_distance_between_sets(X, Y):
    if isinstance(X, list):
        X = np.asarray(X)
        
    if isinstance(Y, list):
        X = np.asarray(Y)
    distances_matrix = np.sqrt(np.sum(np.square(X-Y), axis=1))
    return distances_matrix

def eval(feature_model, data_loader, X_mean_0, X_mean_1):
    feature_model.eval()
    y_gt = []
    y_pre = []
    count = 0
    for i, data in enumerate(tqdm(data_loader)):
        
        img = data[0]
        label = data[1]
        if label == 0:
            continue
            
        with torch.no_grad():
            output = feature_model(img)
            output = output.view(img.size()[0],512,-1)
            output = output.view(-1,output.size()[-1])
            output = F.normalize(output, p=2, dim=1)
            output = output.detach().numpy()
            
            distance_0 = np.mean(average_distance_between_sets(X_mean_0, output))
            
            distance_1 = np.mean(average_distance_between_sets(X_mean_1, output))
            y_gt.append(label.numpy())
            pre = int((distance_0 > distance_1)*1)
            y_pre.append(pre)
        count +=1
        if count >= 30:
            break
    return y_gt, y_pre
            
X_mean_0 = np.load('weights/RestNet50CNNSpot_Feature.npy')
X_mean_1 = np.load('weights/RestNet50CNNSpot_Feature_1.npy')

y_true, y_pred = eval(features, data_loader, X_mean_0, X_mean_1)
y_true = [int(i[0]) for i in y_true]
cm = confusion_matrix(y_true, y_pred)
print(' ')
print(cm)



    
    








