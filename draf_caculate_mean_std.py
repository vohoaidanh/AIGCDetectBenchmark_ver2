import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        self.all_images = self.fake_images# self.real_images + self.fake_images
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# Define the paths to your datasets
real_image_path = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\0_real'
fake_image_path = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\1_fake'

# Define a transform to resize and convert images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the custom dataset
dataset = CustomImageDataset(real_image_path, fake_image_path, transform=transform)

# Create a DataLoader to iterate through the dataset
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)


import torch
from tqdm import tqdm 

def calculate_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images in tqdm(dataloader):
        # batch_samples is the number of images in the batch
        batch_samples = images.size(0)
        n_samples += batch_samples
        
        # Calculate mean and std for each batch and sum them up
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    # Normalize the sum of means and stds by the number of samples
    mean /= n_samples
    std /= n_samples

    return mean, std

if __name__ == '__main__':
    # Calculate mean and std
    mean, std = calculate_mean_std(dataloader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")









