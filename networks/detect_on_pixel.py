import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_shape = (3,16,16)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3*8*8, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(3,-1)
        img = self.model(z)
        #img = img.view(img.size(0), *self.img_shape)
        return img

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 3, 1024)  # Example hidden layer with 64 units
        self.fc2 = nn.Linear(1024, 512)  # Example hidden layer with 32 units
        self.fc3 = nn.Linear(512, 3)  # Output layer with 3 units

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RandomCropDataset(Dataset):
    def __init__(self, root, transform=None):
        
        self.root = root
        self.transform = transform
        self.image_paths = self.__get_imgs_list()
    
    def __get_imgs_list(self):
        result = []
        for a,b,c in os.walk(self.root):
            if len(c)>0:
                imgs = [os.path.join(a,i) for i in c if i.endswith(('jpg', 'png', 'webp', 'jpeg'))]
            result.extend(imgs)
            
        return result
            

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        

        # Ensure the image is big enough for a 3x3 crop
        width, height = img.size
        if width < 3 or height < 3:
            raise ValueError("Image size must be at least 3x3 pixels")

        # Randomly select the top-left corner of the 3x3 crop
        x = random.randint(0, width - 3)
        y = random.randint(0, height - 3)

        # Perform the 3x3 crop
        crop = img.crop((x, y, x + 3, y + 3))

        # Convert crop to tensor
        crop = transforms.ToTensor()(crop)
        crop = (crop-0.5)/0.5

        # Get the 8 pixels (excluding the center)
        input_pixels = torch.cat((crop[:, 0, 0:3], crop[:, 1, 0:1], crop[:, 1, 2:], crop[:, 2, 0:3]), dim=1).view(8, 3)
        
        # Get the center pixel
        center_pixel = crop[:, 1, 1]

        return input_pixels, center_pixel


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=0)

    def forward(self, x1, x2):
        return 1 - self.cosine_similarity(x1, x2).mean()
    
def evaluation(model, dataset): 
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    model.eval()
    total_loss = 0.0
    for data, label in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, label)  # Compute loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)

    print(f"Epoch [{epoch+1}], eval avg_loss: {avg_loss}")
    return avg_loss

from tqdm import tqdm 
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Create the model instance
    
    # Define a transform to resize images to a minimum size if necessary
    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Define a list of image paths (you should replace this with your own image paths)
    image_paths = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset\val\0_real'
    
    # Create the dataset
    dataset = RandomCropDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create the dataset
    eval_dataset = RandomCropDataset(r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset\val\1_fake', transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
       
    model = SimpleModel()
    #criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    criterion = CosineSimilarityLoss()  # Custom cosine similarity loss

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loss = []
    test_loss = []    
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for data, label in tqdm(dataloader):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, label)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        train_loss.append(avg_loss)
        print(f"Epoch [{epoch+1}/{1000}], avg_loss: {avg_loss}")
            
        model.eval()
        total_loss = 0.0
        for data, label in tqdm(eval_loader):
            with torch.no_grad():
                outputs = model(data)  # Forward pass
                loss = criterion(outputs, label)  # Compute loss
                total_loss += loss.item()
        avg_loss = total_loss / len(eval_loader)
        test_loss.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{1000}], eval avg_loss: {avg_loss}")

   

    

gen_model = Generator()
batch_size = 1
input_features = 192
input_tensor = torch.randn(batch_size, input_features)  # Example input tensor

gen_model(input_tensor)









