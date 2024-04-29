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
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
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










