CUDA_VISIBLE_DEVICES=0  PORT=29503 ./tools/dist_test.sh  configs/ASMNet/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e.py \
 work_dirs/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e/epoch_30.pth 1 --format-only --eval-options \
 submission_dir=work_dirs/ASMNet_S_fpn_o-rcnn-dotav1-ss_le90_30e/Task1_results_ss_ASMNet_S_fpn_o-rcnn-dotav1-ss_epoch30

