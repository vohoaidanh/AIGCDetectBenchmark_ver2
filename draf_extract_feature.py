import torch
from torch import nn
from torchvision import transforms
from options import TrainOptions, TestOptions
from data.datasets import read_data_combine
from data import create_dataloader_new,create_dataloader
#from networks.resnet import resnet50, resnet18
import matplotlib.pyplot as plt
#from util import get_model
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from data.process import get_processing_model
import numpy as np
from data.datasets import read_data_new
from tqdm import tqdm
import torch.utils.model_zoo as model_zoo

###############################################################
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

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
################################################################
def get_model(opt):
    if opt.detect_method in ["CNNSpot","LNP","LGrad","DIRE", "Derivative", "CNNSpot_Noise", "Resnet_Mask"]:
        if opt.isTrain:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(model.fc.weight.data, 0.0, opt.init_gain)
            return model
        else:
            return resnet50(num_classes=1)



def validation(model,opt, mask = None):
    from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
    i = 0
    opt = get_processing_model(opt) 
    data_loader = create_dataloader_new(opt)
    y_true, y_pred = [], []
    y_pred_ = []

    if mask is not None:
        layer = model.layer4[2].relu  # Accessing 'layer4.2.relu_2'
        hook = layer.register_forward_hook(get_apply_mask_hook(mask))
    
    with torch.no_grad():
        for img, label in tqdm(data_loader):
            i += 1
            #print("batch number {}/{}".format(i, len(data_loader)), end='\r')
            in_tens = img#.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
            
            y_pred_.extend(model(in_tens).flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred_ = np.array(y_pred_)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.8)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.8)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred>=0.5)
    conf_mat = confusion_matrix(y_true, y_pred_>=15.0)
    if mask is not None:
        hook.remove()
    
    return acc, ap, conf_mat, r_acc, f_acc, y_true, y_pred, list(zip(y_pred_, y_true))


def convert(feature, r = 2**6, c = 2**5, kernel_size=7):
    feature = feature.view(r*c, kernel_size**2)
    feature = feature.reshape(r, c, kernel_size, kernel_size)

    feature = feature.permute(0, 2, 1, 3)
    feature = feature.reshape(r*kernel_size, c * kernel_size)
    return feature

def invert_convert(feature, r=2**6, c=2**5, kernel_size=7):
    # Reshape from (r*kernel_size, c*kernel_size) to (r, c, kernel_size, kernel_size)
    feature = feature.reshape(r, kernel_size, c, kernel_size)
    
    # Permute dimensions back to (r, kernel_size, c, kernel_size)
    feature = feature.permute(0, 2, 1, 3)
    
    # Reshape to (r*c, kernel_size, kernel_size) which is (r*c, 7, 7) in this case
    feature = feature.reshape(r*c, kernel_size, kernel_size)
    
    return feature

# Hàm hook với mask
def apply_mask(module, input, output, mask):
    if output.shape == torch.Size([32, 2048, 7, 7]):
        output = output * mask
    print('output shape', output.shape)
    return output

# Định nghĩa một lớp đóng gói để truyền mask vào hook
def get_apply_mask_hook(mask):
    def hook(module, input, output):
        return apply_mask(module, input, output, mask)
    return hook


def create_mask(model, opt, layer_name='layer4.2.relu_2'):
    opt = get_processing_model(opt) 
    opt.batch_size = 1
    data_loader = create_dataloader_new(opt)
    
    return_nodes = {
        layer_name: 'feature',
    }
    
    feature_extractor = create_feature_extractor(
    	model, return_nodes=return_nodes) # This is last layer of layer2 in resnet
    
    model.eval()
    pos_feature = []
    neg_feature = []
    with torch.no_grad():
        for img, label in tqdm(data_loader):
            if device == torch.device('cuda'):
                in_tens = img.cuda()
            else:
                in_tens = img

            # label = label.cuda()
            output = feature_extractor(in_tens)
            output = convert(output['feature'])
            if label == 1:# Real Image
                pos_feature.append(output)
            else:
                neg_feature.append(output)    
                
    feature = torch.stack(pos_feature, dim=0)
    pos_feature_mean = torch.mean(feature,dim=0)
    
    feature = torch.stack(neg_feature, dim=0)
    neg_feature_mean = torch.mean(feature,dim=0)
    diff = (pos_feature_mean > neg_feature_mean).float()
    mask = invert_convert(diff)
    result = {
        'pos_feature_mean': pos_feature_mean,
        'neg_feature_mean': neg_feature_mean,
        'diff': diff,
        'mask': mask
        }
    
    return result
    
import pickle

if __name__ == '__main__':
        
    dataroot = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'.replace('\\','/')
    val = 'val'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = TestOptions().parse(print_options=True) #获取参数类
    opt.batch_size = 1
    opt.detect_method = 'CNNSpot'
    opt.pos_label = '0_real'
    opt.isTrain = False
    opt.isVal = False
    opt.pos_label = '0_real'
    opt.model_path = r"D:\K32\do_an_tot_nghiep\data\CNNSpot_1isreal_model_epoch_best.pth".replace('\\', '/')
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.process_device = device
    opt = get_processing_model(opt) 
    
    model = get_model(opt)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'],strict=True)
    
    train_nodes, eval_nodes = get_graph_node_names(model)
    return_nodes = {
        # node_name: user-specified key for output dict
        'layer4.2.relu_2': 'layer3',
    }
    
    features = create_mask(model, opt)
    MASK = features['mask']
    
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print(50*'=')
    opt.batch_size = 32
    mask = torch.zeros(MASK.shape)
    result = validation(model=model, opt=opt, mask=(MASK))
    
    
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)
    

    
    
    
    
    
    
