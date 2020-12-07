import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist)
import open3d as o3d

from gradslam.datasets import (build_dataloader, build_dataset)
from gradslam.models import build_slam
from gradslam.core.structures.rgbdimages import RGBDImages

def parse_args():
    parser = argparse.ArgumentParser(
        description='CCSLAM test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_slam(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    item = next(iter(data_loader))

    # SLAM
    pointclouds, recovered_poses = model(item)

    # visualization
    o3d.visualization.draw_geometries([pointclouds.open3d(0)])
    o3d.visualization.draw_geometries([pointclouds.open3d(1)])





if __name__ == '__main__':
    main()