import torch
import torch.nn as nn

__all__ = []

def conv7x7(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

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
        self.unfold = nn.Unfold(kernel_size=64, stride=64)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.layer_norm = nn.LayerNorm((512, 1, 1))

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
        unfolded = self.unfold(x)
        unfolded = unfolded.permute(0, 2, 1)
        unfolded = unfolded.reshape(unfolded.size(0), unfolded.size(1),-1, 64, 64)
        
        out = []
        for x in unfolded: 
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.layer_norm(x)
            x = x.view(x.size(0), -1)
            out.append(x)
        out = torch.stack(out, dim=0)
        out = torch.mean(out, dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    
class CNNBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(CNNBasicBlock, self).__init__()
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv7x7(inplanes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.pool(out1)
        out1 = out1.view(out1.size(0),self.planes)
        out1 = self.relu(out1)
        

        out2 = self.conv2(x)
        out2 = self.pool(out2)
        out2 = out2.view(out2.size(0),self.planes)
        out2 = self.relu(out2)
        
        out3 = self.conv3(x)
        out3 = self.pool(out3)
        out3 = out3.view(out3.size(0),self.planes)
        out3 = self.relu(out3)
        
        out = torch.stack([out1,out2,out3])
        out = torch.mean(out, dim = 0)
        return out

    
class CNNSimpest(nn.Module):

    def __init__(self, patch_size=21, num_classes=1):
        
        super(CNNSimpest, self).__init__()
        self.inplanes = 3
        self.outplans = 512
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=28, stride=27)
        self.layer1 = CNNBasicBlock(self.inplanes, self.outplans, stride=1)
        
        self.head = nn.Sequential(
            nn.Linear(self.outplans, self.outplans*2), 
            nn.ReLU(inplace=False), 
            nn.Dropout(p=0.3),
            nn.Linear(self.outplans*2, num_classes)
        )

    def forward(self, x):
        unfolded = self.unfold(x)
        unfolded = unfolded.permute(0, 2, 1)
        unfolded = unfolded.reshape(unfolded.size(0), unfolded.size(1),-1, self.patch_size, self.patch_size)

        #out = self.layer1(unfolded)
        out = []
        for i in unfolded:
            y = self.layer1(i)
            y = torch.mean(y, dim=0)
            out.append(y)
        out = torch.stack(out, dim=0)
        out = self.head(out)
        return out


def cnn_simpest(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 1], **kwargs)
    return model

if __name__ == '__main__':
    
    #test model
    model = cnn_simpest(num_classes=1)

    input_tensor = torch.randn(2, 3, 112, 112)
    
    out_tensor = model(input_tensor)
 

