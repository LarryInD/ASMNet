_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_30e.py',
    '../_base_/datasets/dotav1_ss1024.py'
]

project_dir = '{{fileBasenameNoExtension}}'  # TODO change
work_dir = f'./work_dirs/{project_dir}'

num_classes = 15
bs = 4  # batch_size
num_workers = bs * 4
base_lr = 0.0001

save_interval = 1
val_interval = 6
max_keep_ckpts = 4  # TODO change

# base_batch_size = (4 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8, enable=False)

checkpoint = ''  # pretrain # noqa
# load_from = ''  # resume # noqa
angle_version = 'le90'

# model settings
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ASMNet_S_ESWA', 
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


# batch_size = (4 GPUs) x (2 samples per GPU) = 8
data = dict(
    samples_per_gpu=bs,
    workers_per_gpu=num_workers,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# evaluation
evaluation = dict(interval=val_interval, metric='mAP', save_best='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=save_interval, max_keep_ckpts=max_keep_ckpts)

