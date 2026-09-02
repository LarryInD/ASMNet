# ASMNet
ASMNet: Axis-decoupled sparse mining for efficient remote sensing object detection

ASMNet.py can be added as a backbone network to, for example, the [PKINet](https://github.com/PKINet/PKINet) repository for training. 

### Pretrained models
pretrained ASMNet-S backbone: [Download](https://1drv.ms/u/c/d050e31ac8f91613/IQCisSD0GRoqTZS_LY3kQ-xFAZ8sxfYn3kl7c9RyljVJ-8k?e=KoGAQk)

pretrained ASMNet-T_NoNaN backbone: [Download](https://1drv.ms/u/c/d050e31ac8f91613/IQCAnTVttV7USJw6SjEV3i75AYomTpG-2bLmwvhhpnoWCs8?e=hZHnbG)


### Training
The configuration files under configs/ASMNet can be added to the configs/ directory of, for example, the [PKINet](https://github.com/PKINet/PKINet) repository for training.

To train the model, use:

CUDA_VISIBLE_DEVICES=6,7 PORT=29504 bash ./tools/dist_train.sh configs/ASMNet/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py 2

Adjust the paths to dist_train.sh and the config file (e.g. ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py) as needed.

