import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = ['ResNet', 'resnet50']


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

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ks=1, ndim=1, norm_type=None, act_cls=None, bias=False):
        super(ConvLayer, self).__init__()
        layers = []
        if norm_type == 'Spectral':
            layers.append(nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=ks, bias=bias)))
        else:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=ks, bias=bias))
        if act_cls is not None:
            layers.append(act_cls())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class SelfAttention(nn.Module):
    "Self-attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query = self._conv(n_channels, n_channels // 8)
        self.key = self._conv(n_channels, n_channels // 8)
        self.value = self._conv(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def _conv(self, n_in, n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type='Spectral', act_cls=None, bias=False)

    def forward(self, x):
        size = x.size()  # (batch_size, n_channels, seq_len)
        x = x.view(*size[:2], -1)  # (batch_size, n_channels, seq_len)
        f = self.query(x)  # (batch_size, n_channels // 8, seq_len)
        #print('f====', f.shape)
        g = self.key(x)  # (batch_size, n_channels // 8, seq_len)
        #print('g====', g.shape)

        h = self.value(x)  # (batch_size, n_channels, seq_len)
        #print('h====', h.shape)

        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)  # (batch_size, seq_len, seq_len)
        o = torch.bmm(h, beta)  # (batch_size, n_channels, seq_len)
        o = self.gamma * o + x  # (batch_size, n_channels, seq_len)
        
        return o.view(*size).contiguous()  # (batch_size, n_channels, seq_len)


class RestAttention(nn.Module):
    def __init__(self, model:ResNet, init_head_weight=True, pretrained=True, inference = False):
        super(RestAttention,self).__init__()
        self.inference = inference
        self.model =  model
        #for name, param in self.model.named_parameters():
        #    param.requires_grad = False
        self.model.to(device)
        self.conv1 = conv1x1(512,64)
        self.conv2 = conv1x1(1024,64)
        self.linear1 = nn.Linear(64*28*28, 64*14*14)

        self.feature_extractor = create_feature_extractor(
        	self.model, return_nodes=['fc', 'layer2.3.relu_2', 'layer3.5.relu_2']) # This is last layer of layer2 in resnet

        self.attn = SelfAttention(n_channels=64)      
        self.attn2 = SelfAttention(n_channels=64) 
                
        self.head = nn.Sequential(
            nn.Linear(64*14*14*2, 2048), 
            nn.ReLU(inplace=False), 
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1)
        )
        
        if init_head_weight:
            for m in self.head:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
        
    def forward(self, x):
           
        _extractor =  self.feature_extractor(x)
        feature_extractor = _extractor['layer2.3.relu_2']
        #label_pre = _extractor['fc']
        feature_extractor = self.conv1(feature_extractor)
        batch_size, c, w, h = feature_extractor.shape
        feature_extractor = feature_extractor.view((batch_size, c, -1))
        #print('feature_extractor========',feature_extractor.shape)
        
         
        feature_extractor2 = self.conv2(_extractor['layer3.5.relu_2'])
        batch_size2, c2, w2, h2 = feature_extractor2.shape
        feature_extractor2 = feature_extractor2.view((batch_size2, c2, -1))
        #print('feature_extractor2========',feature_extractor2.shape)
       

        #print('feature_extractor=====', feature_extractor.shape)
        attn1 = self.attn(feature_extractor)
        attn2 = self.attn2(feature_extractor2)
        
        attn1 = attn1.view((batch_size, -1))
        attn2 = attn2.view((batch_size2, -1))
        attn1 = self.linear1(attn1)
        attn1 = F.softmax(attn1, dim=1)
        #combined_tensor = (attn1 + attn2) / 2        
        combined_tensor = torch.cat((attn1, attn2), dim=1)
        output = self.head(combined_tensor)
        #output = feature_extractor
        return output
    
    def get_parameters_to_optimize(self, target_model_names = ['attn','attn2', 'head', 'conv1','conv2','linear1']):
        parameters_to_update = []
        for name, param in self.named_parameters():
            if any(target_model_name in name for target_model_name in target_model_names):
                parameters_to_update.append(param)
        return parameters_to_update
    

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet_attention():
    resnet = resnet50(num_classes=1, pretrained=False)
    model = RestAttention(resnet)
    return model


if __name__ == '__main__':
    print(32*'=' + '\nruning resnet_attention\n' + 32*'=')
    model  = resnet_attention()
    return_node, _ = get_graph_node_names(model.model)
    in_tensor = torch.rand((2,3,224,224))
    out_tensor = model(in_tensor)
    
# =============================================================================
#     out_tensor.shape
#     
#     for name, param in model.named_parameters():
#         print(f"Name: {name}, Shape: {param.shape}")
# 
# =============================================================================

    


















