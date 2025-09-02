import os
from PIL import Image
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, X_path, Y_path, transform=None):
        self.X_path = X_path
        self.Y_path = Y_path
        self.transform = transform

        self.images_ls = sorted([ f"{X_path}{img}" for img in os.listdir(X_path) if img.endswith(".gif") or img.endswith(".jpg") or img.endswith(".png")])
        self.masks_ls = sorted([ f"{Y_path}{mask}" for mask in os.listdir(Y_path) if mask.endswith(".gif") or mask.endswith(".jpg") or mask.endswith(".png")] )
        
        assert len(self.images_ls) == len(self.masks_ls)
    
    def custom_transform(self, img, mask):
        if self.transform:
            img, mask = self.transform(img), self.transform(mask)
    
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images_ls[index]).convert("RGB")
        mask = Image.open(self.masks_ls[index]).convert("L")

        img, mask = self.custom_transform(img, mask)
        return img, mask
     
    def __len__(self):
        return len(self.images_ls)