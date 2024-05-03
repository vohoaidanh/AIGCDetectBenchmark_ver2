from networks.resnet import resnet50

import torch
from torchvision import transforms

from PIL import Image

model = resnet50(num_classes=1)
features = torch.nn.Sequential(*list(model.children())[:-1])


# Định nghĩa biến biến đổi để chuyển đổi ảnh về kích thước và định dạng phù hợp
transform = transforms.Compose([
    transforms.Resize(224),                      # Chuyển kích thước ảnh về kích thước mong muốn
    transforms.ToTensor(),                              # Chuyển đổi ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Chuẩn hóa giá trị pixel theo mean và std của ImageNet
                         std=[0.229, 0.224, 0.225])
])

input_image = Image.open(r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\1_fake\0d24cf0c-769f-459f-b291-c89effa9283a.jpg")
input_image_tens = transform(input_image)
input_image_tens = input_image_tens.unsqueeze(0)
features.eval()
with torch.no_grad():
    output = features(input_image_tens)

features_output_1d = output.view(1,2048)


import torch
from options import TrainOptions
from data.datasets import read_data_combine
from data import create_dataloader_new,create_dataloader

opt = TrainOptions().parse()
opt.dataroot = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot2 = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.val_split)
opt.batch_size = 1
opt.method_combine = 'CNNSpot+FreDect'

data_loader = create_dataloader_new(opt)
m=None
for i in data_loader:
    print(i)
    m=i
    break


import matplotlib.pyplot as plt

m = i[0].view((3,224,224))
m = m.permute(1,2,0)
plt.imshow(m)

from networks.resnet_combine import resnet50_combine

model = resnet50_combine(pretrained=True, num_classes=1, checkpoint1='resnet50', checkpoint2='resnet50')







assert len([1,2,3]) == len([4,5,6, 7]), \
    "Number of samples in both datasets must be the same."




import os
os.path.basename('dasd/fsdf/afsd/fsdf.jpg')



assert 'a' == 'b', "hello"


from networks.contrastive_resnet.resnet_contrastive import SupConResNet
import torch
from networks.contrastive_resnet.losses import SupConLoss

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a fixed size
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

dataset = ImageFolder(root=r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset', transform=transform)
test_loader = DataLoader(dataset, batch_size=3, shuffle=True)


#with torch.no_grad():
model = SupConResNet(head='linear', feat_dim=2)

   
criterion_model = SupConLoss(temperature=0.001, contrast_mode='one')

features[1] = torch.tensor([100.0,0.0])

for images, labels in test_loader:
    features = model(images)
    features = features.unsqueeze(1) 
    loss = criterion_model(features)
    print(loss.item())
    break


features.shape[1]
contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

anchor_dot_contrast = torch.div(
    torch.matmul(contrast_feature, contrast_feature.T),
    0.1)

logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
logits = anchor_dot_contrast - logits_max.detach()

    
from torch import nn
import torch.nn.functional as F

triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
anchor = contrast_feature[0,:]
anchor = anchor.unsqueeze(0)
anchor = anchor.repeat(3,1)

triplet_loss(anchor,contrast_feature,0.8*contrast_feature)


F.cosine_similarity(anchor, 0.7*contrast_feature, dim=1)

torch.max(anchor_dot_contrast, dim=1, keepdim=True)

anchor.repeat(2,1)
    

torch.scatter(
    torch.ones_like(anchor),
    1,
    torch.arange(1 * 2).view(-1, 1).to('cpu'),
    0
)

z = torch.randn(3, 4)

torch.matmul(z, z.t()) / 4




