pipeline = [
    dict(type='ColorSeqFormatBundle'),
    dict(type='DepthSeqFormatBundle',scaling_factor=5000),
    dict(type='PoseSeqFormatBundle'),
    dict(type='HomoTransformSeqFormatBundle'),
    dict(
        type='Collect',
        meta_keys=('resize','framename','timestamps'),
        keys=['color_seq','depth_seq','intrinsics', 'gt_pose_seq', 'gt_transf_seq']),
]
data_root='/data/TUM/rgbd_dataset_freiburg1_xyz/'
dataset_type='TUMDataset'
data=dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root= data_root,
        ann_files=dict(rgb='rgb.txt',depth='depth.txt',
                       gt_pose='groundtruth.txt'),
        img_prefix='',
        depth_prefix='',
        pipeline=pipeline,
        resize=None,
        seqlen=5,
    )
)