import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, X_path, Y_path, transform=None, img_shape=None):
        self.X_path = X_path
        self.Y_path = Y_path
        self.transform = transform
        self.img_shape = img_shape

        self.images_ls = sorted([ f"{X_path}{img}" for img in os.listdir(X_path) if img.endswith(".gif") or img.endswith(".jpg") or img.endswith(".png")])
        self.masks_ls = sorted([ f"{Y_path}{mask}" for mask in os.listdir(Y_path) if mask.endswith(".gif") or mask.endswith(".jpg") or mask.endswith(".png")] )
        
        assert len(self.images_ls) == len(self.masks_ls)
    
    def custom_transform(self, img, mask, img_shape):
        if self.transform:
            img, mask = self.transform(img), self.transform(mask)
        else:
            transform_img = transforms.Compose([
                transforms.Resize(img_shape),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #            std=[0.229, 0.224, 0.225]),
                transforms.ToTensor()
            ])
            transform_mask = transforms.Compose([
                transforms.Resize(img_shape),
                transforms.ToTensor()
            ])
            img, mask = transform_img(img), transform_mask(mask)

        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images_ls[index]).convert("RGB")
        mask = Image.open(self.masks_ls[index]).convert("L")
        img, mask = self.custom_transform(img, mask, self.img_shape)
        return img, mask
     
    def __len__(self):
        return len(self.images_ls)