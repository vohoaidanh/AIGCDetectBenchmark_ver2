from networks.resnet import resnet50
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from options import TrainOptions
from data.datasets import read_data_combine
from data import create_dataloader_new,create_dataloader
import matplotlib.pyplot as plt
opt = TrainOptions().parse()
opt.dataroot = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.batch_size = 1


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

model = resnet50(pretrained=False,num_classes=1)
state_dict = torch.load(r"D:\K32\do_an_tot_nghiep\data\240428_CNNSpot_checkpoint\model_epoch_best.pth", map_location='cpu')
model.load_state_dict(state_dict['model'])

target_layers = [model.layer4[-1]]

transform = transforms.Compose([
    transforms.Resize(224),                      # Chuyển kích thước ảnh về kích thước mong muốn
    transforms.ToTensor(),                              # Chuyển đổi ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Chuẩn hóa giá trị pixel theo mean và std của ImageNet
                         std=[0.229, 0.224, 0.225])
])



cam = XGradCAM(model=model, target_layers=target_layers)


input_image = Image.open(
    r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\val\0_real\000611338.jpg")
input_image_tens = transform(input_image)
input_image_tens = input_image_tens.unsqueeze(0)
    
input_image_tens = torch.randn(2,3,224,224)

grayscale_cam = cam(input_tensor=input_image_tens)
grayscale_cam = grayscale_cam[0, :]

rgb_img = np.asarray(input_image.resize((grayscale_cam.shape[-1],grayscale_cam.shape[0])))/255.0

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


plt.imshow(visualization)
plt.axis('off')

print(cam.outputs)

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

nodes, _ = get_graph_node_names(model)


feature_extractor = create_feature_extractor(
	model, return_nodes=['fc', 'layer2.3.relu_2'])
# `out` will be a dict of Tensors, each representing a feature map
out = feature_extractor(input_image_tens)
out['layer2.3.relu_2']

features = torch.nn.Sequential(*list(model.children())[:6])
out_f = features(input_image_tens)



torch.sum(out['fc'] - out_f)

import torch.nn as nn


class Footless(nn.Module):
    def __init__(self):
        super(Footless, self).__init__()
        # Load the pre-trained ResNet model
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        # Remove the first three layers (including input layer)
        self.features = nn.Sequential(*list(self.resnet.children())[6:])
        
    def forward(self, x):
        # Forward pass through the remaining layers of ResNet
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x



footless = Footless()

out_2 = footless(out['layer2.3.relu_2'])

for k,i in enumerate(nodes):
    if 'layer2.3.relu_2' == i:
        print(k)

len(nodes)
res_layers = list(model.named_children())
len(res_layers[7])










