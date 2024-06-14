import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from data.process import FourierTransform
import time
from torch.optim.lr_scheduler import StepLR

__all__ = ['resnet18','resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # print(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        # print("1"+str(x.size()))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print("2"+str(x.size()))
        x = self.fc(x)
        # print("3"+str(x.size()))
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
    
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
####################################################################################


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




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, verbose=False, delta=0, model_path='weights/model_best.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta
        self.model_path = model_path

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ...')
        #model.save_networks('best')
        torch.save(model.state_dict(), self.model_path + 'model_best.pth')

        self.score_max = score

class SiameseDataset(Dataset):
    def __init__(self, root, transform=None, pos_label = '1_fake'):
        Extension = ('jpg', 'png', 'jpeg', 'webp', 'tif', 'jpeg')   
        Extension = Extension + tuple([i.upper() for i in Extension])
            
        self.transform = transform
        real_imgs = os.listdir(os.path.join(root, '0_real'))
        real_imgs = [os.path.join(root, '0_real', i) for i in real_imgs if i.endswith(Extension)]
        
        fake_imgs = os.listdir(os.path.join(root, '1_fake'))
        fake_imgs = [os.path.join(root, '1_fake', i) for i in fake_imgs if i.endswith(Extension)]
        
        # pos_label is image class will be choice to anchor and positive pair
        if pos_label == '1_fake':
            real_value, fake_value = 0, 1
            self.pos_imgs, self.neg_imgs = fake_imgs, real_imgs
        else:
            real_value, fake_value = 1, 0
            self.pos_imgs, self.neg_imgs = real_imgs, fake_imgs
            

    def __len__(self):
        return len(self.pos_imgs)
    
    def __split_image(self,image):
        # Mở ảnh
       
        # Lấy kích thước của ảnh
        width, height = image.size
        
        # Tính toán điểm chia (nửa chiều rộng của ảnh)
        mid_point = width // 2
        
        # Cắt ảnh thành phần bên trái và bên phải
        left_image = image.crop((0, 0, mid_point, height))
        right_image = image.crop((mid_point, 0, width, height))
        
        return left_image, right_image
    
    def __crop(self, img, side, crop_size=(256, 256)):

        if side == 'right':
            l, t, r, b = img.width//2, img.height//2, img.width, img.height
            crop_img = img.crop((l, t, r, b))
        
        elif side == 'left':
            l, t, r, b = 0, 0, img.width//2, img.height//2
            crop_img = img.crop((l, t, r, b))
        else:
            crop_img = img
        
        crop_width, crop_height = crop_size
        left = random.randint(0, crop_img.width - crop_width)
        top =  random.randint(0, crop_img.height  - crop_height)
        
        return crop_img.crop((left, top, left + crop_width, top + crop_height))

    def __getitem__(self, idx):
        
        def _resize(im, size=512):
            if min(im.size) < size:
                if  min(im.size) == im.width:
                    new_height = int((size/im.width) * im.height)
                    im = im.resize((size, new_height))
                else:
                    new = int((size/im.height) * im.width)
                    im = im.resize((new, size))
            return im
        
        main_im = Image.open(self.pos_imgs[idx]).convert("RGB")
        main_im = _resize(main_im, 512)

        neg_im = Image.open(random.choice(self.neg_imgs)).convert("RGB")
        neg_im =  _resize(neg_im, 256)
        
        
        anchor_im, pos_im = self.__split_image(main_im)
        #pos_im = Image.open(random.choice(self.pos_imgs)).convert("RGB")

        #anchor_im = self.__crop(pos_im, 'left')
        #pos_im = self.__crop(pos_im, 'right')
        #neg_im = self.__crop(neg_im, side=None)
        
        if self.transform:
            anchor_im = self.transform(anchor_im)
            pos_im = self.transform(pos_im)
            neg_im = self.transform(neg_im)
           
        label = random.choice([0, 1])           
        if label == 1:
            return anchor_im, pos_im, label
        else:
            return anchor_im, neg_im, label

class SiameseDatasetVal(Dataset):
    
    def __init__(self, root, transform=None, pos_label = '1_fake'):
        Extension = ('jpg', 'png', 'jpeg', 'webp', 'tif', 'jpeg')   
        Extension = Extension + tuple([i.upper() for i in Extension])
            
        self.transform = transform
        real_imgs = os.listdir(os.path.join(root, '0_real'))
        real_imgs = [os.path.join(root, '0_real', i) for i in real_imgs if i.endswith(Extension)]
        
        fake_imgs = os.listdir(os.path.join(root, '1_fake'))
        fake_imgs = [os.path.join(root, '1_fake', i) for i in fake_imgs if i.endswith(Extension)]
        
        # pos_label is image class will be choice to anchor and positive pair
        if pos_label == '1_fake':
            real_value, fake_value = 0, 1
            self.pos_imgs, self.neg_imgs = fake_imgs, real_imgs
        else:
            real_value, fake_value = 1, 0
            self.pos_imgs, self.neg_imgs = real_imgs, fake_imgs
            

    def __len__(self):
        return len(self.pos_imgs)
    
    def __split_image(self,image):
        # Mở ảnh
       
        # Lấy kích thước của ảnh
        width, height = image.size
        
        # Tính toán điểm chia (nửa chiều rộng của ảnh)
        mid_point = width // 2
        
        # Cắt ảnh thành phần bên trái và bên phải
        left_image = image.crop((0, 0, mid_point, height))
        right_image = image.crop((mid_point, 0, width, height))
        
        return left_image, right_image
    
    def __crop(self, img, side, crop_size=(256, 256)):

        if side == 'right':
            l, t, r, b = img.width//2, img.height//2, img.width, img.height
            crop_img = img.crop((l, t, r, b))
        
        elif side == 'left':
            l, t, r, b = 0, 0, img.width//2, img.height//2
            crop_img = img.crop((l, t, r, b))
        else:
            crop_img = img
        
        crop_width, crop_height = crop_size
        left = random.randint(0, crop_img.width - crop_width)
        top =  random.randint(0, crop_img.height  - crop_height)
        
        return crop_img.crop((left, top, left + crop_width, top + crop_height))

    def __getitem__(self, idx):
        
        def _resize(im, size=512):
            if min(im.size) < size:
                if  min(im.size) == im.width:
                    new_height = int((size/im.width) * im.height)
                    im = im.resize((size, new_height))
                else:
                    new = int((size/im.height) * im.width)
                    im = im.resize((new, size))
            return im
        
        main_im = Image.open(self.pos_imgs[idx]).convert("RGB")
        main_im = _resize(main_im, 512)

        neg_im = Image.open(random.choice(self.neg_imgs)).convert("RGB")
        neg_im =  _resize(neg_im, 256)
        
        
        anchor_im, _ = self.__split_image(main_im)
        pos_im = Image.open(random.choice(self.pos_imgs)).convert("RGB")

        #anchor_im = self.__crop(pos_im, 'left')
        #pos_im = self.__crop(pos_im, 'right')
        #neg_im = self.__crop(neg_im, side=None)
        
        if self.transform:
            anchor_im = self.transform(anchor_im)
            pos_im = self.transform(pos_im)
            neg_im = self.transform(neg_im)
           
        label = random.choice([0, 1])           
        if label == 1:
            return anchor_im, pos_im, label
        else:
            return anchor_im, neg_im, label


class ContrastiveDataset(Dataset):
    
    def __init__(self, root, transform=None, pos_label = '0_real'):
        root = root.replace('\\', '/')
        Extension = ('jpg', 'png', 'jpeg', 'webp', 'tif', 'jpeg')   
        Extension = Extension + tuple([i.upper() for i in Extension])
            
        self.transform = transform
        real_imgs = os.listdir(os.path.join(root, '0_real'))
        real_imgs = [os.path.join(root, '0_real', i) for i in real_imgs if i.endswith(Extension)]
        
        fake_imgs = os.listdir(os.path.join(root, '1_fake'))
        fake_imgs = [os.path.join(root, '1_fake', i) for i in fake_imgs if i.endswith(Extension)]
        
        # pos_label is image class will be choice to anchor and positive pair
        if pos_label == '1_fake':
            pos_imgs, neg_imgs = fake_imgs, real_imgs
        else:
            pos_imgs, neg_imgs = real_imgs, fake_imgs
                
        self.pos_imgs = self.__filter_imgs(256, pos_imgs)
        self.neg_imgs = self.__filter_imgs(256, neg_imgs)

        self.all_image = self.pos_imgs + self.neg_imgs
        self.all_label = [1 for _ in range(len(self.pos_imgs))] + [0 for _ in range(len(self.neg_imgs))]
            

    def __len__(self):
        return len(self.all_image)
        
    
    def __split_image(self,image):
        # Mở ảnh
       
        # Lấy kích thước của ảnh
        width, height = image.size
        
        # Tính toán điểm chia (nửa chiều rộng của ảnh)
        mid_point = width // 2
        
        # Cắt ảnh thành phần bên trái và bên phải
        left_image = image.crop((0, 0, mid_point, height))
        right_image = image.crop((mid_point, 0, width, height))
        
        return left_image, right_image

    def __filter_imgs(self, minsize = 256, imgs=[]):
        all_label = []
        for path in imgs:
            im = Image.open(path).convert("RGB")
            if min(im.size) < minsize:
                continue
            all_label.append(path)
        return all_label
    
    def __getitem__(self, idx):
                
        def _resize(im, size=512):
            if min(im.size) < size:
                if  min(im.size) == im.width:
                    new_height = int((size/im.width) * im.height)
                    im = im.resize((size, new_height))
                else:
                    new = int((size/im.height) * im.width)
                    im = im.resize((new, size))
            return im
        
        main_im, label = Image.open(self.all_image[idx]).convert("RGB"), self.all_label[idx]
        main_im = _resize(main_im, 512)
        label = torch.tensor(label)
        label = label.float()
        pos_left, pos_right = self.__split_image(main_im)
        neg_im = Image.open(random.choice(self.neg_imgs)).convert("RGB")

        
        if self.transform:
            main_im = self.transform(main_im)
            pos_left = self.transform(pos_left)
            pos_right = self.transform(pos_right)
            neg_im = self.transform(neg_im)
            
        if label==1:
            return main_im, pos_left, pos_right, label
        else:
            return main_im, pos_left, neg_im, label
   
        
        
class NormalDataset(Dataset):
    def __init__(self, root, transform=None, pos_label = '1_fake'):
        root = root.replace('\\', '/')
        Extension = ('jpg', 'png', 'jpeg', 'webp', 'tif', 'jpeg')   
        Extension = Extension + tuple([i.upper() for i in Extension])
            
        self.transform = transform
        real_imgs = os.listdir(os.path.join(root, '0_real'))
        real_imgs = [os.path.join(root, '0_real', i) for i in real_imgs if i.endswith(Extension)]
        
        fake_imgs = os.listdir(os.path.join(root, '1_fake'))
        fake_imgs = [os.path.join(root, '1_fake', i) for i in fake_imgs if i.endswith(Extension)]
        
        # pos_label is image class will be choice to anchor and positive pair
        if pos_label == '1_fake':
            self.pos_imgs, self.neg_imgs = fake_imgs, real_imgs
        else:
            self.pos_imgs, self.neg_imgs = real_imgs, fake_imgs
        
        self.all_image = self.pos_imgs + self.neg_imgs
        self.all_label = [1 for _ in range(len(self.pos_imgs))] + [0 for _ in range(len(self.neg_imgs))]
            

    def __len__(self):
        return len(self.all_image)
    

    def __getitem__(self, idx):
        
        def _resize(im, size=512):
            if min(im.size) < size:
                if  min(im.size) == im.width:
                    new_height = int((size/im.width) * im.height)
                    im = im.resize((size, new_height))
                else:
                    new = int((size/im.height) * im.width)
                    im = im.resize((new, size))
            return im
        
        main_im, label = Image.open(self.all_image[idx]).convert("RGB"), self.all_label[idx]
        main_im = _resize(main_im, 512)

        
        if self.transform:
            main_im = self.transform(main_im)
      
        return main_im, label

# =============================================================================
# class ContrastiveLoss(nn.Module):
#     """
#     Contrastive loss function.
#     """
# 
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
# 
#     def forward(self, x0, x1, y):
#         # euclidian distance
#         diff = x0 - x1
#         dist_sq = torch.sum(torch.pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)
# 
#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]
#         return loss
# =============================================================================
    
class ContrastiveLoss(nn.Module):
    #target = 0 represent dissimilar pairs
    #target = 1 represent similar pairs
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(0.5 * target * torch.pow(euclidean_distance, 2) +
                                      0.5 * (1 - target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class CombinedLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_cls=1.0, lambda_sim=0.3):
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.lambda_cls = lambda_cls
        self.lambda_sim = lambda_sim

        self.contrastive_loss = ContrastiveLoss(margin=self.margin)
        self.bce_loss = nn.BCEWithLogitsLoss()


    def forward(self, cls_output, output1, output2, label):
        loss_contrastive = self.contrastive_loss(output1, output2, label)
        loss_cls = self.bce_loss(cls_output, label)
        loss = self.lambda_sim*loss_contrastive + self.lambda_cls * loss_cls
        
        return loss
    
class SiameseNetwork(nn.Module):

    def __init__(self,embedding_size=128, **kwargs):
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.encoder = resnet50(pretrained=True)
        
        for name, param in self.encoder.named_parameters():
            if not name.startswith('layer4'):
                param.requires_grad = False
        
                
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        self.projector = nn.Sequential(
                nn.Linear(2048, self.embedding_size),
            )
        
        self.cls_fc = nn.Sequential(
                nn.Linear(self.embedding_size, 1),
            )
        
    def get_parameters_to_optimize(self, target_model_names = ['encoder', 'projector', 'cls_fc']):
        parameters_to_update = []
        for name, param in self.named_parameters():
            if any(target_model_name in name for target_model_name in target_model_names):
                parameters_to_update.append(param)
        parameters_to_update = [p for p in parameters_to_update if p.requires_grad == True]
        return parameters_to_update

    def forward_once(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x
    
    def classify(self, x):
         x = self.forward_once(x)
         x = self.cls_fc(x)
         return x.squeeze(1)
    
    def forward(self, input0, input1, input2=None):
        output1 = self.forward_once(input1)
        if input2 is None:
            input2 = input1

        output2 = self.forward_once(input2)        
        output_cls = self.classify(input0)
        
        return output_cls, output1, output2
    
    
def siamese_network(pretrained = False):
    model = SiameseNetwork()
    return model

def train(model, train_loader, val_loader, args=None):
    model_path = 'weights/'
    early_stop =  EarlyStopping(patience=5, verbose=True, delta=0.001, model_path = model_path)
    model.cuda()
    loss=[] 
    counter=[]
    iteration_number = 0
    #criterion = ContrastiveLoss()
    criterion = CombinedLoss(lambda_cls = args.lambda_loss[0], lambda_sim = args.lambda_loss[1])
    optimizer = torch.optim.Adam(model.get_parameters_to_optimize(), lr=1e-3, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    experiment = comet_log(args)

    total_steps = 0
    for epoch in range(1,1000):
        model.train()
        loss = []
        
        start_time = time.time()
        
        for i, data in enumerate(train_loader,0):
            total_steps += 1
            main_im, img0, img1 , label = data
            main_im, img0, img1 , label = main_im.cuda(), img0.cuda(), img1.cuda() , label.cuda()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Load data {:.2f} giây".format(elapsed_time))
            
            optimizer.zero_grad()
            cls_output, output1, output2 = model(main_im, img0, img1)
            
            #print('cls_output ', cls_output.dtype)
            #print('label ', label.dtype)
 
            loss_contrastive = criterion(cls_output, output1, output2, label)
            loss_contrastive.backward()
            loss.append(loss_contrastive.item())
            optimizer.step()    
            
            start_time = time.time()

            
            if total_steps % 5 == 0:
                print("Train loss: {} at step: {}".format(loss_contrastive.item(), total_steps))
            
        print("Epoch {}\n Current loss {}\n".format(epoch, torch.tensor(loss).mean()))
        
        print('validate', 30*'=')
        accuracy, precision, recall, f1, all_labels, distances = validate(model, val_loader, transform)
        print(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        #for item1, item2 in zip(all_labels, distances):
        #    print(f"({item1}, {item2})")
        #print(f'saving model epoch {epoch} ', 30*'=')

        early_stop(score=accuracy, model=model)

        torch.save(model.state_dict(), model_path + 'model_last.pth')

        iteration_number += 10
        counter.append(iteration_number)
        
        scheduler.step()

        ####################################

        if experiment is not None:
            experiment.log_metric('val/epoch_acc', accuracy, epoch=epoch)
            experiment.log_metric('val/precision', precision, epoch=epoch)
            experiment.log_metric('val/recall', recall, epoch=epoch)
            experiment.log_metric('val/f1', f1, epoch=epoch)

            #file_name = "epoch_{}_val_{}.json".format(epoch, comet_train_params['name'])
            #experiment.log_confusion_matrix(matrix = val_conf_mat, file_name=file_name, epoch=epoch)
        ####################################

    return model



def validate(model, dataloader, transform, threshold=0.5):
    model.eval()
    model.cuda()
    iteration_number = 0
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for i, data in enumerate(dataloader,0):
    
            main_im, img0, img1 , label = data
            main_im, img0, img1 , label = main_im.cuda(), img0.cuda(), img1.cuda() , label.cuda()
            
            all_labels.extend(label.cpu().numpy())  
            cls_output, output1, output2 = model(main_im, img0, img1)
            #distances = F.pairwise_distance(output1, output2)
            #preds = (distances < threshold).float()
            preds = cls_output.sigmoid().flatten()
            preds = (preds > 0.5).long()  # torch.Tensor with binary values (0 or 1)
            all_preds.extend(preds.cpu().numpy())
            #.sigmoid().flatten().tolist()
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1, all_labels, all_preds


def inference_one(model, image, pos_im, neg_im):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.cuda()
    with torch.no_grad():
        output1, output2 = model(image,pos_im)
        distances_to_pos = F.pairwise_distance(output1, output2)
        
        output1, output2 = model(image,neg_im)
        distances_to_neg = F.pairwise_distance(output1, output2)
        print(f'neg: {distances_to_neg}; pos: {distances_to_pos}', )
        
    return (distances_to_pos < distances_to_neg)*1

def gen_feature(model, data_path, name = 'weights/features.pkl'):
    import pickle
    from tqdm import tqdm

    fouries_fz = FourierTransform(filter_='highpass', cutoff=0.5)
    transform = transforms.Compose([
        transforms.CenterCrop((256,256)),
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet']),  # Normalize to [-1, 1]
        fouries_fz,
    ])
    
    dataset = NormalDataset(root=data_path,transform=transform)
    data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    
    pos_feature = []
    neg_feature = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    for data, label in tqdm(data_loader):
        with torch.no_grad():
            data = data.to(device)
            out,_ = model(data)
            out = out.cpu().detach().numpy()
            if label==1:
                pos_feature.append(out)
            else:
                neg_feature.append(out)

        
    pos_feature = np.concatenate(pos_feature, axis=0)
    neg_feature = np.concatenate(neg_feature, axis=0)
    
    
    data = {
        'pos_feature': pos_feature,
        'neg_feature': neg_feature,
        }
    
    # Save the data dictionary to disk using pickle
    with open(name, 'wb') as f:
        pickle.dump(data, f)

    
    

    

def load_model(checkpoint_path=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = siamese_network(pretrained=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model            
         


####################################################################################
import comet_ml
def comet_log(opt):
    if opt.comet:
        comet_ml.init(api_key='MS89D8M6skI3vIQQvamYwDgEc')
        comet_train_params = {
            'name of method': 'Siamese_net with combine of Contrastive and BCEwithlogit loss',
            'dataset_name': 'RealFakeDB512s',
            }
        
        experiment = comet_ml.Experiment(
            project_name="ai-generated-image-detection"
        )
        
        experiment.log_parameter('Train params', comet_train_params)
        return experiment
    
    return None

import argparse

if __name__ == '__main__':
    from PIL import Image
    #from torchvision import transforms
    
    parser = argparse.ArgumentParser(description="A script that sets a flag.")
    parser.add_argument('-e', '--comet', action='store_true', help='A flag to enable comet.')
    parser.add_argument('--method', default='train')
    parser.add_argument('--cutoff', type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument('--lambda_loss', type=float, nargs=2, default=[1.0, 0.5], help='lamba_sim / lambda_cls')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cutoff[0] == 0.0 and args.cutoff[1] == 1.0:
        fouries_fz = transforms.Lambda(lambda img: img) # 不处理
        print('fouries_fz = transforms.Lambda(lambda img: img)')
    else:
        fouries_fz = FourierTransform(cutoff=args.cutoff)
        print('fouries_fz = FourierTransform(cutoff=args.cutoff)')
    transform = transforms.Compose([
        transforms.RandomCrop((224,224)),
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet']),  # Normalize to [-1, 1]
        fouries_fz,
    ])


    
    train_dataset = ContrastiveDataset(root=r'dataset/real_gen_dataset/train',
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=4)
    print(f'len of train set: {len(train_loader)}')
    
    val_dataset = ContrastiveDataset(root=r'dataset/RealFakeDB_test/test',
                                   transform=transform)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4)
    
    print(f'len of test set: {len(val_loader)}')

    if args.method == 'train':
        model = siamese_network()
        model.to(device)
        model = train(model, train_loader, val_loader, args)
    elif args.method == 'eval':
        transform = transforms.Compose([
        transforms.CenterCrop((256,256)),
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet'])  # Normalize to [-1, 1]
        ])

        model = load_model(r"weights/model_best.pth")
        val_dataset = ContrastiveDataset(root=r'dataset/real_gen_dataset/val',
                                       transform=transform)
        
        val_loader = DataLoader(val_dataset,
                                batch_size=64,
                                shuffle=False,
                                num_workers=4)
        
        accuracy, precision, recall, f1, all_labels, distances = validate(model, val_loader, transform)
        print(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
 
    elif args.method == 'gen_feature':
        model = load_model(r"weights/model_best.pth")
        gen_feature(model, 
                    data_path = r'/workspace/AIGCDetectBenchmark_ver2/dataset/RealFakeDB/test', 
                    name = 'weights/RealFakeDB_test_feature.pkl')













