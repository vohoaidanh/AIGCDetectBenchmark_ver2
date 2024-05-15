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

    def __init__(self, patch_size=7, num_classes=1):
        
        super(CNNSimpest, self).__init__()
        self.inplanes = 3
        self.outplans = 512
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=7, stride=7)
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
    model = CNNSimpest(**kwargs)
    return model

if __name__ == '__main__':
    
    #test model
    model = cnn_simpest(num_classes=10)

    input_tensor = torch.randn(1, 3, 256, 224)
    
    out_tensor = model(input_tensor)
    out_tensor = out_tensor.view(-1)
   
