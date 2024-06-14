import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.nn.functional as F


__all__ = ['resnet18', 'cosine_similarity_loss']


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


class Resnet_Metric(nn.Module):

    def __init__(self,embedding_size=128, **kwargs):
        super(Resnet_Metric, self).__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_size)
            )
        
    def __crop_input(self,image_tensor):
        # Ensure the input tensor has three dimensions (channel, height, width)
        assert len(image_tensor.shape) == 3, "Input tensor must have shape [C, H, W]"
        
        # Get the dimensions of the input image tensor
        _, height, width = image_tensor.shape
        
        # Calculate the center of the image
        center_height = height // 2
        center_width = width // 2
        
        # Crop the image tensor into four equal parts
        top_left = image_tensor[:, :center_height, :center_width]
        top_right = image_tensor[:, :center_height, center_width:]
        bottom_left = image_tensor[:, center_height:, :center_width]
        bottom_right = image_tensor[:, center_height:, center_width:]
        
        # Return the four cropped image tensors
        idx = np.random.choice(range(0,4), size = 2, replace=False)
        imgs = [top_left, top_right, bottom_left, bottom_right]
        return imgs[idx[0]].unsqueeze(0), imgs[idx[1]].unsqueeze(0)
    
    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        #x1, x2 = self.__crop_input(x)
        x1 = self.encoder(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.projector(x1)
        x2 = self.encoder(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.projector(x2)
        
        x1 = F.normalize(x1, p=2, dim=-1)  # L2 normalization
        x2 = F.normalize(x2, p=2, dim=-1)  # L2 normalization

        cosine_similarity = F.cosine_similarity(x1, x2)
        return (cosine_similarity.unsqueeze(1) + 1.0)/2.0
    
    def get_parameters_to_optimize(self, target_model_names = ['encoder', 'projector']):
        parameters_to_update = []
        for name, param in self.named_parameters():
            if any(target_model_name in name for target_model_name in target_model_names):
                parameters_to_update.append(param)
        return parameters_to_update
    

    
    def get_parameters_to_optimize(self, target_model_names = ['encoder', 'projector']):
        parameters_to_update = []
        for name, param in self.named_parameters():
            if any(target_model_name in name for target_model_name in target_model_names):
                parameters_to_update.append(param)
        return parameters_to_update

def resnet_metric(pretrained = False):
    model = Resnet_Metric()
    if pretrained:
        model.resnet18.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def cosine_similarity_loss(cosine_sim, target, margin=0.0):
  
    # Adjust target from 0 to -1 for dissimilar pairs
    adjusted_target = 2 * target - 1  # Convert 1 to 1, and 0 to -1
    cosine_sim = cosine_sim * 2.0 - 1.0
    # Compute the loss 
    loss = (1 - adjusted_target) * F.relu(cosine_sim - margin) + adjusted_target * (1.0 - cosine_sim)
    return torch.mean(loss)



if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    
    model = resnet_metric(pretrained=False)

    pars = model.get_parameters_to_optimize()    
    
    img = Image.open(r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\val\1_fake\6e59c694-14cd-4e7a-a3c3-992c5e044dcc.jpg")
    
    transform = transforms.Compose([
        transforms.Resize(224),                      # Chuyển kích thước ảnh về kích thước mong muốn
        transforms.ToTensor(),                              # Chuyển đổi ảnh thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Chuẩn hóa giá trị pixel theo mean và std của ImageNet
                             std=[0.229, 0.224, 0.225])
    ])
    
    x = transform(img)
    x = x.unsqueeze(0)
    out = model(x)
    print(out)
   
