import cv2
import yaml
import torch
import glob
import numpy as np
from src.models.backbones import *
from src.models.heads import *


def resize_with_height(img, ratio, ori_w, target_height):
        new_w = int(ori_w * ratio)
        img = cv2.resize(img, (new_w, target_height))
        return img, new_w
    
def resize_with_width(img, ratio, ori_h, target_width):
    new_h = int(ori_h * ratio)
    img = cv2.resize(img, (target_width, new_h))
    return img, new_h
    

def resize_with_pad(img, h=256, w=256):
    ori_h, ori_w = img.shape[:2]
    top, bottom, left, right = 0, 0, 0, 0

    if ori_h >= ori_w:
        ratio = h / ori_h
        img, new_w = resize_with_height(img, ratio, ori_w, h)
        pad = w - new_w
        left = pad // 2
        right = pad - left
        
    else:
        ratio = w / ori_w
        img, new_h = resize_with_width(img, ratio, ori_h, w)
        pad = h - new_h
        top = pad // 2
        bottom = pad - top
        
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)
    return img


cfg_path = 'configs/generalization_cfg.yml'
cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
model_cfg = cfg['Model']
pretrain_cfg = cfg['Pretrain']

backbone_model = Resnet(**model_cfg['ImgBackbone'])
teacher_head = Classification(**model_cfg['Teacher_Head'])

checkpoint = torch.load(pretrain_cfg['checkpoint_path'])
backbone_model.load_state_dict(checkpoint['backbone_state_dict'])
teacher_head.load_state_dict(checkpoint['teacher_state_dict'])


def return_CAM(feature_conv, weight, class_idx):
    """
    return_CAM generates the CAMs and up-sample it to 224x224
    arguments:
    feature_conv: the feature maps of the last convolutional layer
    weight: the weights that have been extracted from the trained parameters
    class_idx: the label of the class which has the highest probability
    """
    size_upsample = (224, 224)
    
    # we only consider one input image at a time, therefore in the case of 
    # VGG16, the shape is (1, 512, 7, 7)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))# -> (512, 49)
        cam = np.matmul(weight[idx], beforeDot) # -> (1, 512) x (512, 49) = (1, 49)
        cam = cam.reshape(h, w) # -> (7 ,7)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

img_path_list = glob.glob('DANH/real_gen_dataset/test/1_fake/*.*')

for idx, img_path in enumerate(img_path_list[:10]):
    img_path = 'DANH/real_gen_dataset/train/real/000609337.jpg'
    name = img_path.split('/')[-1]
    ori_img = cv2.imread(img_path)
    img = ori_img.astype(np.float32) / 255.0
    img = resize_with_pad(img)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img)[None, :]

    feature_cov = backbone_model(img)
    out = teacher_head(feature_cov)[0]
    print(">>>>>>>>>>>>>>>>>")
    print(idx)
    print(out)
    params = list(teacher_head.h_fc.parameters())
    weight = np.squeeze(params[0].data.numpy())

    CAMs = return_CAM(feature_cov.detach().numpy(), weight, [0])
    heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)

    # image = ori_img.reshape((256, 256, 3))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = image * std + mean
    # image = np.clip(image, 0, 1)
    # print(heatmap.shape)
    # exit()
    img = cv2.resize(ori_img, (256, 256))
    heatmap = cv2.resize(heatmap, (256, 256))
    result = 0.5 * heatmap + 0.5 * img
    cv2.imwrite(f'debugs/CAM/{idx}.png', result)





