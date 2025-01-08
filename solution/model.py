import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageClassificationDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, transform=None):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.transform = transform
        self.positive_files = [(os.path.join(positive_dir, f), 1) for f in os.listdir(positive_dir) if f.endswith('.png')]
        self.negative_files = [(os.path.join(negative_dir, f), 0) for f in os.listdir(negative_dir) if f.endswith('.png')]
        self.all_files = self.positive_files + self.negative_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path, label = self.all_files[idx]
        image = Image.open(file_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, os.path.basename(file_path)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: 2 classes (positive and negative)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()