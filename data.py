# Imports
import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

tran = transforms.Compose([transforms.RandomCrop(256), transforms.ToTensor(), normalize])

# Data Class
class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms=tran):
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'
        if not (os.path.exists(content_dir_resized) and os.path.exists(style_dir_resized)):
            os.mkdir(content_dir_resized)
            os.mkdir(style_dir_resized)
            self._resize(content_dir, content_dir_resized)
            self._resize(style_dir, style_dir_resized)
        
        content = glob.glob((content_dir_resized + '/*'))
        np.random.shuffle(content)
        style = glob.glob((style_dir_resized + '/*'))
        np.random.shuffle(style)
        self.images_pairs = list(zip(content, style))
        self.transforms = transforms

# Get data
def get_data(content_dir, style_dir, batch_size, train):
  dataset = PreprocessDataset(train_content_dir, train_style_dir)
  return DataLoader(dataset, batch_size=args.batch_size, shuffle=train), len(dataset)
  