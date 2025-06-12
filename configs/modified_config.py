auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
]
data_prefix = dict(img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        sweeps_num=10,
        test_mode=True,
        type='LoadPointsFromMultiSweeps'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.001
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            max_voxels=(
                30000,
                40000,
            ),
            point_cloud_range=[
                -50,
                -50,
                -5,
                50,
                50,
                3,
            ],
            voxel_size=[
                0.25,
                0.25,
                8,
            ])),
    pts_backbone=dict(
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN2d'),
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        anchor_generator=dict(
            custom_values=[
                0,
                0,
            ],
            ranges=[
                [
                    -49.6,
                    -49.6,
                    -1.80032795,
                    49.6,
                    49.6,
                    -1.80032795,
                ],
                [
                    -49.6,
                    -49.6,
                    -1.74440365,
                    49.6,
                    49.6,
                    -1.74440365,
                ],
                [
                    -49.6,
                    -49.6,
                    -1.68526504,
                    49.6,
                    49.6,
                    -1.68526504,
                ],
                [
                    -49.6,
                    -49.6,
                    -1.67339111,
                    49.6,
                    49.6,
                    -1.67339111,
                ],
                [
                    -49.6,
                    -49.6,
                    -1.61785072,
                    49.6,
                    49.6,
                    -1.61785072,
                ],
                [
                    -49.6,
                    -49.6,
                    -1.80984986,
                    49.6,
                    49.6,
                    -1.80984986,
                ],
                [
                    -49.6,
                    -49.6,
                    -1.763965,
                    49.6,
                    49.6,
                    -1.763965,
                ],
            ],
            reshape_out=True,
            rotations=[
                0,
                1.57,
            ],
            sizes=[
                [
                    4.60718145,
                    1.95017717,
                    1.72270761,
                ],
                [
                    6.73778078,
                    2.4560939,
                    2.73004906,
                ],
                [
                    12.01320693,
                    2.87427237,
                    3.81509561,
                ],
                [
                    1.68452161,
                    0.60058911,
                    1.27192197,
                ],
                [
                    0.7256437,
                    0.66344886,
                    1.75748069,
                ],
                [
                    0.40359262,
                    0.39694519,
                    1.06232151,
                ],
                [
                    0.48578221,
                    2.49008838,
                    0.98297065,
                ],
            ],
            type='AlignedAnchor3DRangeGenerator'),
        assigner_per_size=False,
        bbox_coder=dict(code_size=9, type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        feat_channels=384,
        in_channels=384,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=10,
        test_cfg=dict(
            max_num=500,
            min_bbox_size=0,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            use_rotate_nms=True),
        train_cfg=None,
        type='Anchor3DHead',
        use_direction_classifier=True),
    pts_middle_encoder=dict(
        in_channels=64, output_shape=[
            400,
            400,
        ], type='PointPillarsScatter'),
    pts_neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN2d'),
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_strides=[
            1,
            2,
            4,
        ]),
    pts_voxel_encoder=dict(
        feat_channels=[
            64,
            64,
        ],
        in_channels=4,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=[
            -50,
            -50,
            -5,
            50,
            50,
            3,
        ],
        type='HardVFE',
        voxel_size=[
            0.25,
            0.25,
            8,
        ],
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True),
    test_cfg=dict(
        pts=dict(
            max_num=500,
            min_bbox_size=0,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            use_rotate_nms=True)),
    train_cfg=None,
    type='MVXFasterRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            20,
            23,
        ],
        type='MultiStepLR'),
]
point_cloud_range = [
    -50,
    -50,
    -5,
    50,
    50,
    3,
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LidarDet3DInferencerLoader',
                use_dim=5),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=24)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                type='LoadPointsFromMultiSweeps'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                rot_range=[
                    -0.3925,
                    0.3925,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -50,
                    -50,
                    -5,
                    50,
                    50,
                    3,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -50,
                    -50,
                    -5,
                    50,
                    50,
                    3,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ],
                type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(backend_args=None, sweeps_num=10, type='LoadPointsFromMultiSweeps'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            -50,
            -50,
            -5,
            50,
            50,
            3,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -50,
            -50,
            -5,
            50,
            50,
            3,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ],
        type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(save_dir=None, type='LocalVisBackend'),
    ])
voxel_size = [
    0.25,
    0.25,
    8,
]
