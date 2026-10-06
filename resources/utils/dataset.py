import os
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

#---------------------------------- Dataset ----------------------------------
#-----------------------------------------------------------------------------

class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'] + '.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = int(row['label'])
        return image, label

####----------------------------- Transform ---------------
def get_transform(IMG_SIZE = (450, 600)): 
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),  # lesions have no canonical orientation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),  # mild crop only
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                              std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

####----------------------------- Sampler ---------------
def get_sampler(df):
    class_counts = df['label'].value_counts().sort_index().values
    
    class_weights = 1.0 / class_counts  # inverse frequency
    
    # per-sample weight = weight of its class
    sample_weights = df['label'].map(lambda c: class_weights[c]).values
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # required — lets minority classes be resampled
    )
    return sampler 
