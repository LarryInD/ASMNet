# ASMNet
ASMNet: Axis-decoupled sparse mining for efficient remote sensing object detection
## Introduction 👋

This repository contains the implementations of **ASMNet**  for remote sensing object detection.



## Setup 🛠️

MMRotate-PKINet depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMEngine](https://github.com/open-mmlab/mmengine).

For more details, please refer to the [MMRotate installation guide](https://mmrotate.readthedocs.io/en/latest/install.html).

```shell
conda create --name ASMNet python=3.8 -y
conda activate ASMNet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install yapf==0.40.1
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmengine
git clone https://github.com/LarryInD/ASMNet.git
cd ASMNet
mim install -v -e .
pip install timm
```


### Pretrained models
pretrained ASMNet-S backbone: [Download](https://1drv.ms/u/c/d050e31ac8f91613/IQCisSD0GRoqTZS_LY3kQ-xFAZ8sxfYn3kl7c9RyljVJ-8k?e=KoGAQk)

pretrained ASMNet-T_NoNaN backbone: [Download](https://1drv.ms/u/c/d050e31ac8f91613/IQCAnTVttV7USJw6SjEV3i75AYomTpG-2bLmwvhhpnoWCs8?e=hZHnbG)

### Training 🚀

To train the model, use:

CUDA_VISIBLE_DEVICES=6,7   PORT=29503   bash  ./tools/dist_train.sh   configs/ASMNet/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py  2

or
```bash
bash train_ASMNet.sh
```
Adjust the paths to dist_train.sh and the config file (e.g. ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py) as needed. Absolute paths are also perfectly fine if you prefer.


### Test and Submit the result to [DOTA-v1.0 server](http://bed4rs.net:8001/login/?next=/evaluation1/) 🚀
To test the model, use:

CUDA_VISIBLE_DEVICES=7  PORT=29503 ./tools/dist_test.sh  configs/ASMNet/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py \
 work_dirs/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e/epoch_30.pth 1 --format-only \
 --eval-options submission_dir=work_dirs/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e/Task1_results_ss_ASMNet_S_fpn_o-rcnn-dotav1-ss_epoch30


or
```bash
bash test_ASMNet.sh
```


Please adjust the paths to dist_test.sh, the config file (e.g., ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py), the checkpoint path (work_dirs/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e/epoch_30.pth), and the output directory (work_dirs/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e/) as needed. Absolute paths are also perfectly fine if you prefer.
