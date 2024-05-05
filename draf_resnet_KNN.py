# -*- coding: utf-8 -*-
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import euclidean_distances


from networks.resnet import resnet50
import torch
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image

from options import TrainOptions
from data.datasets import read_data_combine
from data import create_dataloader_new,create_dataloader
import numpy as np

import matplotlib.pyplot as plt

def average_pairwise_distance(X, Y):
    # Tính ma trận khoảng cách giữa các điểm từ hai tập hợp
    distances_matrix = pairwise_distances(X, Y)
    # Tính trung bình của ma trận khoảng cách
    avg_distance = np.mean(distances_matrix)
    return avg_distance
# Tính trung bình khoảng cách giữa các điểm trong hai tập hợp vector
def average_distance_between_sets(X, Y):
    if isinstance(X, list):
        X = np.asarray(X)
        
    if isinstance(Y, list):
        X = np.asarray(Y)
        
    distances_matrix = np.sqrt(np.sum(np.square(X-Y), axis=1))
    return distances_matrix
    


# Load dữ liệu dataset

model = resnet50(num_classes=1)
state_dict = torch.load(r"D:\K32\do_an_tot_nghiep\data\240428_CNNSpot_checkpoint\model_epoch_best.pth", map_location='cpu')
model.load_state_dict(state_dict['model'])
features = torch.nn.Sequential(*list(model.children())[:8])

opt = TrainOptions().parse()
opt.dataroot = r'D:\K32\do_an_tot_nghiep\data\RealFakeDB512s'
#opt.dataroot = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot2 = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.val_split)
opt.batch_size = 1
opt.method_combine = None
opt.isTrain = True
opt.isVal = False
opt.noise_type = 'None'
opt.num_threads = 0

data_loader = create_dataloader_new(opt)
features.eval()
X_mean = np.load('weights/RestNet50CNNSpot_Feature.npy')


X = None
count = 0

t1 = 0

for i, data in enumerate(tqdm(data_loader)):
    
    img = data[0]
    label = data[1]
    if torch.sum(label) == 0:
        continue
    count += 1
    with torch.no_grad():
        output = features(img)
        output = output.view(img.size()[0],512,-1)
        output = output.view(-1,output.size()[-1])
        output = F.normalize(output, p=2, dim=1)
        output = output.detach().numpy()
        if X is None:
            X = output
        else:
            X = np.concatenate((X, output), axis=0)
            
    #pre = average_distance_between_sets(X_mean, output)
    #print('\n',np.sum(pre), ":", label)
    #t1 += pre
    if (count%10 == 0):
        print(f"count {count}")
    if count >= 200:
        break
    
np.mean(t1)

#np.save(f'dataset_{opt.train_split.npy}',X)


distances = []
for i in range(len(X)):
    distances.append(euclidean_distances([X_mean[i]], [X[i]])[0][0]
)
    
distances2 = []
for i in range(len(X)):
    distances2.append(euclidean_distances([X_mean[i]], [X[i]])[0][0]
)

plt.bar(range(512), np.asarray(distances[:]))
plt.title('cross label = 1')
plt.show()

plt.bar(range(30), np.asarray(distances2[:30]))
plt.title('label = 1')
plt.show()

np.mean(distances)
np.mean(distances2)

X_mean = []
for i in range(512):
    X_i = X[i::512,:]
    mean = np.mean(X_i, axis=0)
    X_mean.append(mean)

X_mean = np.asarray(X_mean)

plt.figure(figsize=(10, 8))
idx = np.random.choice(range(0,512),4)
#idx = np.array([704, 625, 538, 890]) -512
for k in range(0,512-4,4):
    idx = [i for i in range(k+1,k+5)]
    plt.figure(figsize=(10, 8))
    plt.ioff()
    for i,j in enumerate(idx):
        plt.subplot(2, 2, i+1)
        categories = np.array(range(len(X_mean[j])))
        plt.bar(categories,X_mean[j])
        plt.title(f'Chanel {j}')
    
    plt.savefig(f'images/Resnet50_layer8_c{idx[0]}_{idx[-1]}.png')

np.save('weights/RestNet50CNNSpot_Feature_1.npy', X_mean)

plt.figure(figsize=(10, 8))
# Lặp qua các subplot và vẽ dữ liệu

idx = np.random.choice(range(0,512),4)
idx = np.array([704, 625, 538, 890]) -512
for i,j in enumerate(idx):
    plt.subplot(2, 2, i+1)
    categories = np.array(range(len(X[j])))
    plt.bar(categories,X[j])
    plt.title('C_{}; Real {}/{} ; Pre {}'.format(j, label[0],label[1],(model(data[0]).sigmoid().flatten() > 0.5)*1))
    
#idx = np.random.choice(range(512,1024),4)
for i,j in enumerate(idx+512):
    plt.subplot(2, 2, i+1)
    plt.bar(categories,X[j],alpha=0.5)
    #plt.title('ch_{}'.format(j))


np.sum(output[1,:]**2)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = output.detach().numpy()

np.min(X)

X_train, X_test, y_train, y_test = train_test_split(X, y=None, test_size=0.2, random_state=42)



kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Trực quan hóa kết quả
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', edgecolors='k')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()






transformer = transforms.ToPILImage()
im1 =data[0][0].squeeze(0)
im1 =  transformer(im1)

plt.imshow(im1)

im2 =data[0][1].squeeze(0)
im2 =  transformer(im2)
plt.imshow(im2)







import torch
import torch.nn.functional as F

# Giả sử tensor_image là tensor của bạn có kích thước (1, 3, 256, 256)
tensor_image = torch.randn(1, 3, 256, 256)  # Đây là tensor ngẫu nhiên, bạn cần thay thế bằng dữ liệu thực tế

# Chuẩn hóa tensor theo các chiều cụ thể (ở đây là chiều kênh)
tensor_image_normalized = F.normalize(tensor_image, p=2, dim=1)

# Kiểm tra phạm vi giá trị sau khi chuẩn hóa
min_value = tensor_image_normalized.min().item()
max_value = tensor_image_normalized.max().item()
print("Giá trị nhỏ nhất sau khi chuẩn hóa:", min_value)
print("Giá trị lớn nhất sau khi chuẩn hóa:", max_value)



tensor_image_normalized[0,2,:,:].mean()



X_mean = np.load('weights/RestNet50CNNSpot_Feature.npy')



import numpy as np

# Tính trung bình khoảng cách giữa các điểm trong hai tập hợp vector


# Ví dụ:
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[2, 3], [4, 5]])

avg_distance = average_distance_between_sets(X, Y)


