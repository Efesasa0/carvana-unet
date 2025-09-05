# [Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge) [UNet](https://arxiv.org/abs/1505.04597) Segmentation Using Pytorch

Implementation of the U-Net architecture trained on the Carvana Image masking dataset. Trained on single nvidia A100 PCIe GPU. Bellow are the results with plain settings using [train_val.py](train_val.py) i.e. no weight decay, scheduler, data augmentation.
<table>
    <tr>
        <td style="text-align: center;" colspan="2">
            <img src="assets/figure.png" alt="train loss" width="250">
            <p>Task Summary 64x64 images</p>
        </td>
    </tr>
    <tr>
        <td style="text-align: center;">
            <p>Train Losses</p>
            <img src="assets/train_loss.png" alt="train loss" width="350">
        </td>
        <td style="text-align: center;">
            <p>Validation Losses</p>
            <img src="assets/val_loss.png" alt="validation loss" width="350">
        </td>
    </tr>
</table>

## Getting Started

```bash
    git clone git@github.com:Efesasa0/carvana-unet.git
    cd carvana-unet
    pip install -r requirements.txt
    python train_val.py
```

### Dataset

Downloaded from [kaggle page](https://www.kaggle.com/c/carvana-image-masking-challenge) for the Carvana dataset. Organized  as follows under `./data` directory. `./manual_test` and  `manual_test_masks` consists of only few I specifically selected for fast sanity checks.

```bash
.
├── manual_test'
│   ├── 0cdf5b5d0ce1_01.jpg
│   ├── ...
│   └── 0cdf5b5d0ce1_05.jpg
├── manual_test_masks
│   ├── 0cdf5b5d0ce1_01_mask.gif
│   ├── ...
│   └── 0cdf5b5d0ce1_05_mask.gif
├── train
└── train_masks
```

### References

- [UNet Paper](https://arxiv.org/abs/1505.04597)
- [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)

#### Additional References

- <https://github.com/yakhyo/unet-pytorch/tree/main>
- <https://github.com/asanakoy/kaggle_carvana_segmentation?tab=readme-ov-file>
