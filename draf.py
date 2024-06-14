from networks.resnet import resnet50, resnet18
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from PIL import Image

model = resnet50(num_classes=1)


fc_weights = model.fc.weight.data

fc_weights = fc_weights.numpy()
fc_weights = fc_weights.T
fc_weights[fc_weights<0.0] = 0.0
plt.plot(fc_weights)

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
opt.dataroot = r'D:\dataset\real_gen_dataset'
opt.dataroot2 = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.val_split)
opt.batch_size = 1
opt.method_combine = None#'CNNSpot+FreDect'
opt.detect_method = 'Resnet_Metric'
opt.isTrain

data_loader = create_dataloader_new(opt)
m=None
for i in data_loader:
    #print(i)
    m=i
    break

from PIL import Image
from torchvision import transforms
from networks.resnet_metric import resnet_metric
model = resnet_metric(pretrained=False)

out = model(m[0])
out = out.unsqueeze(1)
out = out.squeeze(1)
pars = model.get_parameters_to_optimize()    

img = Image.open(r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\val\1_fake\6e59c694-14cd-4e7a-a3c3-992c5e044dcc.jpg")

transform = transforms.Compose([
    transforms.Resize(224),                      # Chuyển kích thước ảnh về kích thước mong muốn
    transforms.ToTensor(),                              # Chuyển đổi ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Chuẩn hóa giá trị pixel theo mean và std của ImageNet
                         std=[0.229, 0.224, 0.225])
])

x = transform(img)
out = model(x)
print(out)

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
for i in range(np.random.randint(4,6)):
   x = np.random.randint(0,w)
   y = np.random.randint(0,h)
   strength = np.random.randint(5,10)
   radius = np.random.randint(w//3,w//2)
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

            
  
            

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Assuming 'output' is the output of your model
output = 0.8  # Example output value

probability = sigmoid(output)
print("Probability:", probability)







import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import swirl, warp
from PIL import Image
import copy
# Đọc ảnh đầu vào
import torch
from torchvision import transforms

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from networks.resnet import resnet50


transform = transforms.Compose([
    transforms.Resize(224),                      # Chuyển kích thước ảnh về kích thước mong muốn
    transforms.ToTensor(),                              # Chuyển đổi ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Chuẩn hóa giá trị pixel theo mean và std của ImageNet
                         std=[0.229, 0.224, 0.225])
])

import os
from random import choice
label = ['Fake', 'Real']
root = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset\val\0_real'
root = root.replace('\\', r'/')
img_list = []
for a,b,c in os.walk(root):
    if len(c)>0:

        ims = [os.path.join(a,i) for i in c if i.endswith(('jpg', 'png', 'webp', 'tif'))]
        img_list.extend(ims)
        

model = resnet50(pretrained=False, num_classes=1)

for name, layer in model.named_children():
    print(name)
    
for name, layer in model.named_modules():
    print(name)
    
status_dict = torch.load(r"D:\K32\do_an_tot_nghiep\data\CNNSpot_1isreal_model_epoch_best.pth", map_location=torch.device('cpu'))
model.load_state_dict(status_dict['model'])
target_layers = [model.layer3[-1]]
cam = HiResCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]


input_image = Image.open(choice(img_list))
input_image = input_image.resize((224,224))
input_image_tens = transform(input_image)
input_image_tens = input_image_tens.unsqueeze(0)

input_tensor = input_image_tens# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(np.asarray(input_image, dtype='float32')/255.0, grayscale_cam, use_rgb=True)

# You can also get the model outputs without having to re-inference
model_outputs = int((cam.outputs.sigmoid()>0.5)*1)
plt.imshow(visualization)
plt.axis('off')
plt.title(label[model_outputs])


res18 = resnet18()



plt.imshow(input_image)





status_dict['model']['layer4.2.bn3.weight'].shape

torch.min(status_dict['model']['layer4.2.bn3.running_mean'])



chr(0x3a)




import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, patch_size):
        super(MyModel, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Chia ảnh thành các patch
        unfolded = self.unfold(x)
        
        # Sắp xếp lại shape
        unfolded = unfolded.permute(0, 2, 1).reshape(-1, x.size(1), self.patch_size, self.patch_size)
        
        return unfolded

# Sử dụng mô hình
patch_size = 3
model = MyModel(patch_size)
input_tensor = torch.randn(1, 3, 6, 6)  # Tensor đầu vào với kích thước (c, w, h)
output_tensor = model(input_tensor)
print(output_tensor.size())  # Kích thước của đầu ra


unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
a = unfold(input_tensor)
a = a.permute(0, 2, 1)

a = a.reshape(-1, input_tensor.size(1), 3, 3)

image = Image.open(r'images/dog.jpg')
image.size[0] > 611

import os
os.path.dirname(r'a/images/dog.jpg')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im = Image.open(r"D:\dataset\real_gen_dataset\val\0_real\000613338.jpg")
im1 = im.crop((0,0,224,224))
im2 = im.crop((224,224,224*2,224*2))



from data.process import processing_DCT
opt.dct_mean = torch.load('./weights/auxiliary/dct_mean').permute(1,2,0).numpy()
opt.dct_var = torch.load('./weights/auxiliary/dct_var').permute(1,2,0).numpy()
im_dct1 = processing_DCT(im1, opt)
im_dct2 = processing_DCT(im2, opt)

plt.imshow(im_dct1.permute(1,2,0))
plt.imshow(im_dct2.permute(1,2,0))

opt.dct_mean.shape

import torch.nn.functional as F

anchor = torch.rand((1,100))
contrast_feature = torch.rand((1,100))
F.cosine_similarity(anchor, contrast_feature, dim=1)

a = torch.tensor([[0.0, 1.0]])
b = torch.tensor([[1.0, -1.0]])
F.cosine_similarity(a, b, dim=1)

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.all_images = self.real_images + self.fake_images
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.quantize().convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.dirname(img_path), os.path.basename(img_path)
    

real_image_path = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset/val/0_real'
fake_image_path = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset/val/1_fake'

dataset = CustomImageDataset(real_image_path, fake_image_path)


transform = transforms.Compose([
    #transforms.RandomCrop(224),
    transforms.Resize(5),                      # Chuyển kích thước ảnh về kích thước mong muốn
    transforms.Resize(256),                      # Chuyển kích thước ảnh về kích thước mong muốn

])
 

#img = Image.open(r'D:\Downloads\image/dog2.jpg').convert('RGB')
img, label, _ = dataset[np.random.randint(0, len(dataset))]
transform(img)
print('image is: ', label[-4:])

  
# Importing Image module from PIL package  
from PIL import Image, ImageOps  
import PIL  
import numpy as np

# creating a image object (main image)  
im1= Image.open(r'images/dog.jpg').convert('RGB')
im1 = im1.resize((10,10))
im1 = im1.resize((512,512))
im1_arr = np.asarray(im1)




import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.all_images = self.real_images + self.fake_images
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.quantize().convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.dirname(img_path), os.path.basename(img_path)
    

real_image_path = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset/val/0_real'
fake_image_path = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset/val/1_fake'

dataset = CustomImageDataset(real_image_path, fake_image_path)


from tqdm import tqdm
for i in tqdm(range(len(dataset))):
    image, img_path, img_name = dataset[i]
    new_dir = img_path.replace('real_gen_dataset', 'real_gen_dataset_quantize')
    os.makedirs(new_dir, exist_ok = True)
    image.save(os.path.join(new_dir, img_name))


import torch
from einops import rearrange


# Tạo một tensor PyTorch có hình dạng (2, 3, 4)
a = torch.randn(2, 12)

# Thay đổi hình dạng tensor bằng rearrange
b = rearrange(a, 'x (y z) -> x y z', y=2, z=6)



import torch

# Create a 3x3 tensor
output = torch.rand((3,2,2))
output = torch.mean(output, dim=0).unsqueeze(0)


# Reshape it to add a batch dimension (to match the typical usage in deep learning)
output = output.unsqueeze(0)  # Shape: (1, 3, 3)

# Define window_size
window_size = 4

# Roll the tensor
rolled_output = torch.roll(output, shifts=(1,1), dims=(1, 2))

print(rolled_output)




# Assuming your tensor is named 'tensor'
tensor = torch.randn(3, 5, 5)  # Example random tensor

# Extract the first channel
first_channel = tensor[0:1, :, :]

# Extend the first channel to create a new tensor with all channels having similar values
extended_tensor = torch.cat([first_channel] * 3, dim=0)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Hàm chia ảnh thành các mảnh nhỏ (patches)
def split_and_reconstruct_image(image, grid_size = 4):

    img_width, img_height = image.size
    rows, cols = grid_size, grid_size
    patch_width = img_width // cols
    patch_height = img_height // rows
    patches = []

    for row in range(rows):
        for col in range(cols):
            left = col * patch_width
            upper = row * patch_height
            right = left + patch_width
            lower = upper + patch_height
            box = (left, upper, right, lower)
            patch = image.crop(box)
            patches.append(patch)
    
    new_order = np.random.permutation(len(patches))  # Thay đổi thứ tự sắp xếp nếu bạn muốn

    rows = []
    for i in range(0, len(new_order), grid_size):
        row_patches = [patches[idx] for idx in new_order[i:i + grid_size]]
        row = np.hstack([np.asarray(patch) for patch in row_patches])
        rows.append(row)
    new_image_array = np.vstack(rows)
    return Image.fromarray(new_image_array)



# Đọc ảnh
image_path = r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\0_real\000609935.jpg"# Thay bằng đường dẫn tới ảnh của bạn
image = Image.open(image_path).convert('RGB')

a = split_and_reconstruct_image(image, 64)



image256 = image.quantize(64).convert('RGB')
image256 = image256.convert('RGB')
image256 = image256.convert('L')
image256 = split_and_reconstruct_image(image256, 200)


image3 = np.asarray(image, dtype='float32') - np.asarray(image256, dtype='float32')
image3 = (image3 - np.min(image3))
image3 = np.asarray(image3, dtype='uint8')

fourier_transform = np.fft.fft2(image256)
fourier_transform = np.fft.fftshift(fourier_transform)  # Dịch chuyển zero frequency component đến trung tâm
magnitude_spectrum = np.log1p(fourier_transform)  # Áp dụng log(1 + x) để giữ giá trị dương
magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())  # Chuẩn hóa về phạm vi [0, 1]
magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)  # Chuyển đổi về phạm vi [0, 255] và kiểu uint8


# Hiển thị hình ảnh gốc và hình ảnh sau biến đổi Fourier
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image256, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Fourier Transform')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.show()

image3 = np.asarray(image3, dtype='uint8')
Image.fromarray(image3)
# =============================================================================
# 
# # Custom transformation to apply Fourier Transform to each channel of an RGB image
# class FourierTransform:
#     def __init__(self, radius=2):
#         self.radius = radius
#         
#     def __call__(self, img):
#         if isinstance(img, Image.Image):
#             img = transforms.functional.to_tensor(img)  # Convert PIL image to PyTorch tensor
#         
#         radius = self.radius
#         # Split the image into R, G, B channels
#         r_channel = img[0, :, :]
#         g_channel = img[1, :, :]
#         b_channel = img[2, :, :]
# 
#         # Apply Fourier Transform to each channel
#         r_fourier = torch.fft.fftshift(torch.fft.fft2(r_channel))
#         g_fourier = torch.fft.fftshift(torch.fft.fft2(g_channel))
#         b_fourier = torch.fft.fftshift(torch.fft.fft2(b_channel))
#         
#         _, rows, cols = img.shape
#         crow, ccol = rows // 2, cols // 2  # Center of the image
#         
#         x, y = np.meshgrid(np.arange(cols), np.arange(rows))
#         center = cols//2
#         mask = np.zeros_like(r_channel)
#         mask[(x - center)**2 + (y - center)**2 >= radius**2] = 1.0
#         r_fourier = r_fourier * mask
#         g_fourier = g_fourier * mask
#         b_fourier = b_fourier * mask
# 
#         r_img_back = np.fft.ifft2(r_fourier)
#         g_img_back = np.fft.ifft2(g_fourier)
#         b_img_back = np.fft.ifft2(b_fourier)
#         
#         img_back = torch.stack([torch.tensor(np.abs(r_img_back)), torch.tensor(np.abs(g_img_back)), torch.tensor(np.abs(b_img_back))])
#         img_back = img_back.permute(1,2,0).numpy()
#         img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
#         img_back = img_back.astype('uint8')
# 
#         return Image.fromarray(img_back)
# =============================================================================

# Define the transformations
fourie_fc = FourierTransform(100.0)

fc = transforms.Lambda(lambda img: fourie_fc(img))


transform = transforms.Compose([
    transforms.Resize(5),
    transforms.Resize((256, 256)),
    #fc,
    #transforms.ToTensor(),
    #FourierTransform(),
])

t = transform(image)

t
float(4)
plt.imshow(t.permute((1,2,0)))

t.max()



x, y = np.meshgrid(np.arange(20), np.arange(20))
center = 10
radius = 5
mask = np.zeros((20, 20))
mask[(x - center)**2 + (y - center)**2 >= radius**2] = 1
import cv2


img_filtered_pil = t.permute(1,2,0).numpy()
img_filtered_scaled = cv2.normalize(img_filtered_pil, None, 0, 255, cv2.NORM_MINMAX)
img_filtered_scaled = img_filtered_scaled.astype('uint8')

img_filtered_scaled = Image.fromarray(img_filtered_scaled)

import numpy as np

input = torch.randn(1, 1, 2, 2)
downsample = nn.Conv2d(1, 1, 2, stride=2, padding=1)
upsample = nn.ConvTranspose2d(1, 1, 4, stride=8, padding=1)
h = upsample(input)
h.size()




torch.pow(torch.tensor(3),2)



torch.clamp(torch.tensor(3.2), min=0.0)
torch.sqrt(torch.tensor(2.0))

import pickle 

file_path = r"C:\Users\danhv\Downloads\labels.pkl"

# Open the file in binary read mode and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)


gt = data['all_labels']
pre = data['all_preds']

gt = np.asanyarray(gt, dtype='float32')
pre = np.asanyarray(pre, dtype='float32')

diff = gt==pre


y = np.ones(10), np.ones(10)

y = np.concatenate(y)



import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifft2, fftfreq, ifftshift
import matplotlib.pyplot as plt
from PIL import Image

def create_low_pass_filter_mask(shape, radius):
    """Create a circular low-pass filter mask with given radius."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # Center of the image
    mask = np.zeros((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    center = (y - crow)**2 + (x - ccol)**2
    mask[center <= radius**2] = 1
    return mask

# Load and normalize image
image_path = 'images/dog.jpg'
image_path2 = r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\1_fake\0de9795d-b5d1-4dc6-976f-3677e18cbe19.jpg"
image = Image.open(image_path).convert('L')  # Convert to grayscale
image2 = Image.open(image_path2).convert('L')  # Convert to grayscale
image2 = image2.resize(image.size)
image_array = np.array(image, dtype=np.float32)
image_array2 = np.array(image2, dtype=np.float32)

# Normalize the image to range [0, 1]
image_normalized = image_array / 255.0
image_normalized2 = image_array2 / 255.0

# Apply 2D FFT
transformed_image = fft2(image_normalized)
transformed_image_shifted = fftshift(transformed_image)
phase_spectrum = np.angle(transformed_image_shifted)
magnitude_spectrum = np.abs(transformed_image_shifted)

# Apply 2D FFT
transformed_image2 = fft2(image_normalized2)
transformed_image_shifted2 = fftshift(transformed_image2)
phase_spectrum2 = np.angle(transformed_image_shifted2)
magnitude_spectrum2 = np.abs(transformed_image_shifted2)


plt.imshow(phase_spectrum, cmap='gray')
plt.imshow(np.log(magnitude_spectrum+1e-9), cmap='gray')

#r = np.random.uniform(np.min(magnitude_spectrum), np.max(magnitude_spectrum), magnitude_spectrum.shape)
r  = np.mean(magnitude_spectrum)
#reconstructed_fft_image_shifted = r * np.exp(1j * phase_spectrum)
reconstructed_fft_image_shifted = magnitude_spectrum * np.exp(1j * 1.0)
transformed_image_invert = ifft2(ifftshift(reconstructed_fft_image_shifted))
plt.imshow(transformed_image_invert.real, cmap='gray')

row, col = transformed_image_shifted.shape
mask = create_low_pass_filter_mask(transformed_image_shifted.shape, 30.0)

transformed_image_shifted = transformed_image_shifted*(mask)

plt.imshow(np.log(np.abs(((1-mask).T + 1e-9))), cmap='gray')

plt.imshow(np.log(np.abs(transformed_image_invert) + 0.001), cmap='gray')

plt.imshow(magnitude_spectrum2, cmap='gray')



import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.process import *
from PIL import Image

# =============================================================================
# # Đọc hình ảnh và chuyển đổi sang không gian màu LAB
# image = cv2.imread('images/dog.jpg')
# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# 
# # Tách các kênh L, A, B
# L, A, B = cv2.split(image_lab)
# #L = np.random.randint(112,208,A.shape, dtype=np.uint8)
# L = np.ones_like(L)*180
# # Áp dụng khử nhiễu chỉ trên kênh L
# L_denoised = cv2.fastNlMeansDenoising(L, None, 30, 7, 21)
# # Kết hợp lại các kênh LAB
# image_lab_denoised = cv2.merge((L, A, B))
# 
# # Chuyển đổi trở lại không gian màu BGR
# image_denoised = cv2.cvtColor(image_lab_denoised, cv2.COLOR_LAB2BGR)
# 
# # Hiển thị hình ảnh trước và sau khi khử nhiễu
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')
# 
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(image_denoised, cv2.COLOR_BGR2RGB))
# plt.title('Denoised Image')
# plt.axis('off')
# 
# plt.show()
# 
# 
# plt.imshow(L, cmap='gray')
# np.max(L)
# 
# =============================================================================

if __name__ == '__main__':
    ft = FourierTransform(filter_='highpass', cutoff=0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ft,
        transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.ToTensor(),
        
    ])
        
    
    im = Image.open('images/dog.jpg')

    im_transformed = transform(im)

    im_transformed = im_transformed.permute((1,2,0))
    
    plt.imshow(im_transformed*200.0)    

im_transformed.min()

a = torch.rand((5,3,5))


a[:] = 2.0



