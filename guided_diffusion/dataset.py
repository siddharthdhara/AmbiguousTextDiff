import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TextGuidedImageSegDataset(Dataset):
    """Dataset for text-guided ambiguous image segmentation."""
    
    def __init__(
        self,
        image_paths,
        mask_paths,
        text_annotations,
        image_size=128,
        class_cond=False,
        transform=None,
        return_dict=False,
        image_channels=1, # This should be the controlling parameter
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.text_annotations = text_annotations
        self.image_size = image_size
        self.class_cond = class_cond
        self.return_dict = return_dict
        self.image_channels = image_channels # Store the number of channels
        
        if transform is None:
            
            # Create transform based on the number of channels
            
            norm_mean = [0.5] * self.image_channels
            norm_std = [0.5] * self.image_channels
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ])
        else:
            self.transform = transform
            
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        # Convert image based on the specified number of channels
        if self.image_channels == 1:
            image = Image.open(image_path).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Load mask - handle multiple masks for ambiguity
        mask_path = self.mask_paths[idx]
        
        # If mask_path is a list (multiple annotations), select one randomly
        if isinstance(mask_path, list):
            mask_path = random.choice(mask_path)
            
        # Load and process the mask
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        
        # Get corresponding text annotation
        text = self.text_annotations[idx]
        
        # Optionally return class label for class-conditional models
        if self.class_cond:
            # Use a simple fixed encoding for demonstration
            # In practice, you might have class labels or derive them from the text
            out_dict = {
                "image": image,
                "mask": mask,
                "text": text,
                "y": torch.tensor([0], dtype=torch.long),  # Placeholder class label
            }
        else:
            out_dict = {
                "image": image,
                "mask": mask,
                "text": text,
            }
            
        if self.return_dict:
            return out_dict
        else:
            return image, mask, text