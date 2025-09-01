import torch
import torch.nn as nn
from model_parts import DownSample, UpSample, DoubleConv

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feats=(64, 128, 256, 512)):
        super(Unet, self).__init__()
        
        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        # Down sampling
        for feat in feats:
            self.down_sample.append(DownSample(in_channels, feat))
            in_channels = feat #last iteration --> feats[-1]

        # Bottle neck
        self.bottle_neck = DoubleConv(feats[-1], feats[-1]*2)

        # Up sampling
        for feat in feats[::-1]:
            self.up_sample.append(UpSample(feat*2, feat))

        # Final conv
        self.finalconv = nn.Conv2d(feats[0], out_channels, kernel_size=1)
            
    def forward(self, x):
        h, w = x.shape[-2:]
        assert h % (2 ** len(self.down_sample)) == 0 and w % (2 ** len(self.down_sample)) == 0
        skips_ls = []

        for down_step in self.down_sample:
            skip_x, x = down_step(x)
            skips_ls.append(skip_x)
        
        x = self.bottle_neck(x)
        skips_ls = skips_ls[::-1]

        for idx, up_step in enumerate(self.up_sample):
            x = up_step(x ,skips_ls[idx])
        
        return self.finalconv(x)

"""if __name__ == "__main__":

    x = torch.randn(2,3,160,160) #B, C, H, W
    model = Unet(in_channels=3, out_channels=10, feats=[64, 128, 256])
    preds = model(x)

    print(preds.shape, x.shape)
    print(model)"""