exp_name = 'reds_davsr_25frames_3iter_sf544'

from davsr.datasets.pipelines.augmentation import GenerateUVSRSegmentIndices
from davsr.models.restorers.davsr_model import DAVSRMODEL
from davsr.models.backbones.sr_backbones.davsr_net import DAVSRNet

# model settings
model = dict(
    type=DAVSRMODEL,
    generator=dict(
        type=DAVSRNet,
        n_iter=3,  
        h_nc=16,   
        in_nc=4,
        out_nc=3,
        nc=[16, 16, 16, 16],
        nb=2,
        act_mode="R",                 
        upsample_mode="convtranspose", 
        downsample_mode="strideconv",  
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained='/cluster/work/cvl/videosr/pretrained_models/spynet/spynet_20210409-c6c1bd09.pth',
        vsr_pretrained='/cluster/work/cvl/videosr/pretrained_models/BasicVSR_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth',
        cpu_cache_length=2,  # 100
        sf=(5,4,4),
        interpolation_mode='slomo',
        slomo_pretrained='/cluster/work/cvl/videosr/pretrained_models/superSlomo/SuperSloMo.ckpt',
        ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    )

# model training and testing settings
train_cfg = dict(fix_iter=-1) # we have pretrained it.
test_cfg = dict(metrics=['PSNR','SSIM'], crop_border=2, num_frame_testing=10)

# dataset settings
train_dataset_type = 'SRREDSOrigMultipleGTDataset' 
val_dataset_type = 'SRREDSOrigMultipleGTDataset'  
test_dataset_type = 'SRREDSMultipleGTDataset'

train_pipeline = [
    dict(type=GenerateUVSRSegmentIndices, interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(type=GenerateUVSRSegmentIndices, interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]


test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

real_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(
        type='Collect',
        keys=['lq'],
        meta_keys=['lq_path', 'key'])
]


demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data_dir = 'data'
data = dict(
    workers_per_gpu=12,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, dist=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=f'{data_dir}/REDS_mmedit/train_blur_bicubic_with_val/X4',
            gt_folder=f'{data_dir}/REDS_mmedit/train_orig',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    # val
    val=dict(
        type=test_dataset_type,
        lq_folder=f'{data_dir}/REDS4/blur_bicubic/',
        gt_folder=f'{data_dir}/REDS4/GT/',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
    # val=dict(
    #     type=val_dataset_type,
    #     lq_folder=f'{data_dir}/REDS_mmedit/train_blur_bicubic_with_val/X4',
    #     gt_folder=f'{data_dir}/REDS_mmedit/train_orig',
    #     num_input_frames=5,
    #     pipeline=val_pipeline,
    #     scale=4,
    #     val_partition='REDS4',
    #     test_mode=True),
    # test
    # test=dict(
    #     type=test_dataset_type,
    #     lq_folder=f'{data_dir}/REDS4/blur_bicubic/',
    #     gt_folder=f'{data_dir}/REDS4/GT/',
    #     num_input_frames=10,
    #     pipeline=test_pipeline,
    #     scale=4,
    #     val_partition='REDS4',
    #     test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder=f'{data_dir}/REDS_mmedit/train_blur_bicubic_with_val/X4',
        gt_folder=f'{data_dir}/REDS_mmedit/train_orig',
        num_input_frames=100,
        pipeline=val_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=2500, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=2500, save_image=True, gpu_collect=True) #   remove gpu_collect if not use dist train
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

run_dir = './work_dirs'
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'{run_dir}/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
use_static_graph = True
test_checkpoint_path = f'{run_dir}/{exp_name}/latest.pth' # use --checkpoint None to enable this path in testing
