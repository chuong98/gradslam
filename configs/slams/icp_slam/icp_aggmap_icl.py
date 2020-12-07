_base_='../../base/datasets/icl.py'
dataset=dict(
    train=dict(
        resize=(120,160),
    )
)
model=dict(
    type='ICPSLAM',
    odom_cfg=dict(type="GradICPOdometry",dsratio = 4, numiters = 20, damp = 1e-8, 
                    dist_thr = None, lambda_max = 2.0, 
                    B = 1.0, B2 = 1.0, nu = 200.0),
    map_cfg=dict(type='AggregateMap', inplace=True),
)
train_cfg=None
test_cfg=None