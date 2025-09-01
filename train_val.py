import torch
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from model_main import Unet
from  custom_dataset import CustomDataset

IMG_SHAPE = (64,64)
X_PATH = "../data/train/"
Y_PATH = "../data/train_masks/"
X_TEST_PATH = "../data/manual_test/"
Y_TEST_PATH = "../data/manual_test_masks/"
MODEL_SAVE_PATH = "./models/unet.pth"
FEATS = [64, 128, 256, 512, 1024]

LEARNING_RATE = 3e-3
BATCH_SIZE = 32
EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.ToTensor()
    ])

train_dataset = CustomDataset(X_path=X_PATH, Y_path=Y_PATH, transform=transform)
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

model = Unet(in_channels=3, out_channels=1, feats=FEATS).to(device=device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_running_loss = 0
    for idx, img_mask in enumerate(tqdm(train_dataloader)):
        img, mask = img_mask
        img = img.float().to(device)
        mask = mask.float().to(device)
        
        y_pred = model(img)
        optimizer.zero_grad()
        loss = criterion(y_pred, mask)
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / idx+1

    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader)):
            img, mask = img_mask
            img = img.float().to(device)
            mask = mask.float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)

            val_running_loss += loss.item()
        val_loss = val_running_loss / idx+1
    
    print(f"Ep:{epoch+1} Train_loss:{train_loss:.4f} Val_loss:{val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)