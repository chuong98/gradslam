import gradslam as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import ICL
from gradslam.slam import PointFusion
from torch.utils.data import DataLoader
import argparse
device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icl_path", type=str)
    parser.add_argument("--seqlen", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--show", type=bool, default=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    icl_path = args.icl_path
    dataset = ICL(icl_path, seqlen=args.seqlen, width=args.width, height=args.height)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    colors, depths, intrinsics, poses, *_ = next(iter(loader))
    rgbdimages = RGBDImages(colors, depths, intrinsics)
    slam = PointFusion(device=device)
    pointclouds = Pointclouds(device=device)

    batch_size, seq_len = rgbdimages.shape[:2]
    initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
    prev_frame = None

    for s in range(seq_len):
        live_frame = rgbdimages[:, s].to(device)
        if s == 0 and live_frame.poses is None:
            live_frame.poses = initial_poses
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)
        prev_frame = live_frame

    if args.show:
        pointclouds.plotly(0, max_num_points=15000).update_layout(autosize=False, width=600).show()