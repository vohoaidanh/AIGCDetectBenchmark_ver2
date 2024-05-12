import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
###############################################################################
#https://github.com/htn274/CAN-SupCon-IDS.git
###############################################################################
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import copy

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
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


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

###############################################################################

class Footless(nn.Module):

    def __init__(self, start_layer=7, pretrained=True):
        super(Footless, self).__init__()
        # Load the pre-trained ResNet model
        self.features = resnet50(pretrained=pretrained)
        self.features.fc = nn.Identity()
        # Remove the first three layers (including input layer)
        self.features = nn.Sequential(*list(self.features.children())[start_layer:])
        
    def forward(self, x):
        # Forward pass through the remaining layers of ResNet
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
class LCAM(nn.Module):
    def __init__(self, model:ResNet, init_head_weight=True, pretrained=True, inference = False):
        super(LCAM,self).__init__()
        self.inference = inference
        self.model =  model
        self.footless = Footless(pretrained=pretrained, start_layer=7)
        self.cam_model = resnet50(pretrained=pretrained)
        #for name, param in self.model.named_parameters():
        #    param.requires_grad = False
        self.model.to(device)
        self.footless.to(device)
        self.cam_model.to(device)    
        
        self.feature_extractor = create_feature_extractor(
        	self.model, return_nodes=['fc', 'layer3.5.relu_2']) # This is last layer of layer2 in resnet
        
        self.target_layers = [self.model.layer2[-1]] # is === layer2.3.relu_2 

        #self.shading = resnet50(checkpoint=kwargs['checkpoint2'])
        
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

        self.cam_model.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(2048 * 2, 2048), 
            nn.ReLU(inplace=False), 
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1)
        )
        if init_head_weight:
            for m in self.head:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
        
    def forward(self, x, x_ref):
        #xi is GRB image origin
        #x2 is shading image
        # Forward pass through the first ResNet-50 model
        if x_ref is None:
            x_ref = x
            
        _extractor =  self.feature_extractor(x)
        feature_extractor = _extractor['layer3.5.relu_2']
        label_pre = _extractor['fc']
        layer_feature = self.footless(feature_extractor)
        
        # Forward pass through the second ResNet-50 model
        x_ref = x_ref.to(device)
        grayscale_cam = self.cam(x_ref)
        grayscale_cam = torch.Tensor(grayscale_cam)
        grayscale_cam = grayscale_cam.unsqueeze(1)
        grayscale_cam = grayscale_cam.expand(-1, 3, -1, -1)
        grayscale_cam = grayscale_cam.to(device)
        cam_feature = self.cam_model(grayscale_cam)
        
        # Kết hợp hai features từ hai nhánh
        features  = torch.cat((layer_feature, cam_feature), dim=1)
        output = self.head(features)
        if self.inference:
            label_pre = label_pre.sigmoid()
            label_pre[label_pre > 0.5] = 1.0
            label_pre[label_pre <= 0.5] = 0.0
            label_pre = label_pre.view(output.shape)
            
            output = output.sigmoid()
            output[output > 0.5] = 1.0
            output[output <= 0.5] = 0.0
            output = output*label_pre
        return output
    
    def get_parameters_to_optimize(self, target_model_names = ['footless', 'cam_model', 'head']):
        parameters_to_update = []
        for name, param in self.named_parameters():
            if any(target_model_name in name for target_model_name in target_model_names):
                parameters_to_update.append(param)
        return parameters_to_update
    
def resnet_CAM(model, pretrained=True, inference=False):
    resnet_cam_model = LCAM(model, pretrained=pretrained, inference=inference)
    return resnet_cam_model

    
        
# =============================================================================
# modelresnet = resnet50()
# lcam = LCAM(modelresnet)
# 
# out_3 =  lcam(input_image_tens)
# lcam.return_nodes
# 
# lcam.cam_model
# tensor = torch.Tensor(grayscale_cam)
# tensor = tensor.unsqueeze(1)  # Chiều thứ hai được thêm vào giữa chiều đầu tiên (batchsize) và các chiều còn lại (w, h)
# tensor = tensor.expand(-1, 3, -1, -1)  # -1 để giữ nguyên kích thước của các chiều không thay đổi, 3 là số lượng kênh
# 
# pars = lcam.get_parameters_to_optimize(['cam_model', 'footless', 'head'])
# 
# pars1 = lcam.name_parameters()
# pars1 = [name for name, _ in lcam.named_parameters()]
# 
# ('a', 'b') in ['a','b','c']
#     
# lcam.model
# lcam.footless
# lcam.cam_model
# 
# 
# out_4 = lcam(i[0], i[1])
# =============================================================================
