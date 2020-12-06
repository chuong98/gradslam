pipeline = [
    dict(type='ColorSeqFormatBundle'),
    dict(type='DepthSeqFormatBundle',scaling_factor=5000),
    dict(type='PoseSeqFormatBundle'),
    dict(type='HomoTransformSeqFormatBundle'),
    dict(
        type='Collect',
        keys=['color_seq','depth_seq','intrinsics']),
]
data_root='/data/ICL/living_room_traj1_frei_png/'
dataset_type='ICLDataset'
data=dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root= data_root,
        ann_file=data_root + 'associations.txt',
        pose_file=data_root + 'livingRoom1n.gt.sim',
        img_prefix='',
        depth_prefix='',
        pipeline=pipeline,
        resize=None,
        seqlen=4,
    )
)