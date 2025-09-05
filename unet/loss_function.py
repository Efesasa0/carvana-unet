import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth)
        
        return bce_loss + dice_loss