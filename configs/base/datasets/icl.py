pipeline = [
    dict(type='ColorSeqFormatBundle'),
    dict(type='DepthSeqFormatBundle',scaling_factor=5000),
    dict(type='PoseSeqFormatBundle'),
    dict(type='HomoTransformSeqFormatBundle'),
    dict(
        type='Collect',
        keys=['color_seq','depth_seq','intrinsics', 'gt_pose_seq', 'gt_transf_seq']),
]
data_root='/data/ICL/living_room_traj1_frei_png/'
dataset_type='ICLDataset'
data=dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root= data_root,
        ann_files=dict(associate_file='associations.txt',
                       gt_pose='livingRoom1n.gt.sim'),
        img_prefix='',
        depth_prefix='',
        pipeline=pipeline,
        resize=None,
        seqlen=5,
    )
)