from networks.resnet import resnet50

import torch
from torchvision import transforms

from PIL import Image

model = resnet50(num_classes=1)
layer = list(model.children())
features = torch.nn.Sequential(*list(model.children())[:6])

for i in layer[:-5]:
    print(i.__class__.__name__)
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
opt.dataroot = r'E:\RealFakeDB512'
opt.dataroot2 = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.val_split)
opt.batch_size = 8
opt.method_combine = None#'CNNSpot+FreDect'
opt.detect_method = 'CNNSpot_CAM'
opt.isTrain

data_loader = create_dataloader_new(opt)
m=None
for i in data_loader:
    #print(i)
    m=i
    break


import matplotlib.pyplot as plt

m = i[0][3].view((3,224,224))
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

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

img = Image.open(r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\0_real\000609025.jpg")
img_tensor = transforms.ToTensor()(img)

kernel_x = torch.tensor([[0, 0, 0],
                         [0,-1, 1],
                         [0, 0, 0]], dtype=torch.float32)
kernel_x = kernel_x.unsqueeze(0).repeat(1, 3, 1, 1)



kernel_y = torch.tensor([[0, 0, 0],
                         [0,-1, 0],
                         [0, 1, 0]], dtype=torch.float32)
kernel_y = kernel_y.unsqueeze(0).repeat(1, 3, 1, 1)

output_x = F.conv2d(img_tensor.unsqueeze(0), kernel_x, stride=1)
output_y = F.conv2d(img_tensor.unsqueeze(0), kernel_y, stride=1)

grad_magnitude = torch.sqrt(output_x**2 + output_y**2)
grad_magnitude = grad_magnitude.squeeze(0,1).repeat(3,1,1)

img_grad = transforms.ToPILImage()(grad_magnitude)

output_pil = transforms.ToPILImage()(img_grad)


import matplotlib.pyplot as plt

plt.imshow(img_grad)




 


noise_data = np.random.randint(0, 256, size=(256,256,3), dtype=np.uint8)  # Tạo dữ liệu nhiễu ngẫu nhiên từ 0 đến 255
noise_image = Image.fromarray(noise_data) 



import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import swirl, warp
from PIL import Image
import copy
# Đọc ảnh đầu vào
image = Image.open('images/dog.jpg')
# Thiết lập các tham số cho hiệu ứng liquify
strength = 1  # Độ mạnh của hiệu ứng
radius = 100    # Bán kính của vùng bóp méo

# Áp dụng hiệu ứng liquify bằng hàm swirl từ scikit-image
w, h = image.size
image  = np.asarray(image)

liquified_image = copy.deepcopy(image)
for i in range(np.random.randint(10,15)):
   x = np.random.randint(0,w)
   y = np.random.randint(0,h)
   strength = np.random.randint(1,3)
   radius = np.random.randint(30,100)
   liquified_image = swirl(liquified_image, rotation=0, strength=strength, radius=radius, center=(x,y))

    


# Hiển thị ảnh gốc và ảnh đã được liquify
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(liquified_image)
plt.title('Liquified Image')
plt.axis('off')

plt.show()

            
  
            











