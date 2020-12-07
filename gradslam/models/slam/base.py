from typing import Optional

import torch
import torch.nn as nn

from ..builder import build_odometry, build_map

from gradslam.core.structures import Pointclouds, RGBDImages


class BaseSLAM(nn.Module):
    r"""BASE-SLAM for batched sequences of RGB-D images.

    """
    def __init__(
        self,
        odom_cfg,
        map_cfg,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.odom_cfg = odom_cfg
        self.gt_odom = odom_cfg['type']=='GroundTruthOdometry'
        self.odom = build_odometry(odom_cfg)
        self.map = build_map(map_cfg)
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg

    def forward(self, seq_metas, color_seq, depth_seq,intrinsics, gt_pose=None, **kwargs):
        r"""Builds global map pointclouds from a batch of input RGBDImages with a batch size
        of :math:`B` and sequence length of :math:`L`.

        Args:
            frames (gradslam.RGBDImages): Input batch of frames with a sequence length of `L`.

        Returns:
            tuple: tuple containing:

            - pointclouds (gradslam.Pointclouds): Pointclouds object containing :math:`B` global maps
            - poses (torch.Tensor): Poses computed by the odometry method

        Shape:
            - poses: :math:`(B, L, 4, 4)`
        """
        if isinstance(color_seq,list):
            color_seq = torch.cat(color_seq,dim=0)
        if isinstance(depth_seq,list):
            depth_seq = torch.cat(depth_seq,dim=0)

        device = color_seq.get_device() if torch.cuda.is_available() else torch.device("cpu")
        frames= RGBDImages(color_seq,depth_seq,intrinsics,gt_pose)
        pointclouds = Pointclouds(device=device)
        batch_size, seq_len = frames.shape[:2]
        recovered_poses = torch.empty(batch_size, seq_len, 4, 4).to(device)
        prev_frame = None
        for s in range(seq_len):
            live_frame = frames[:, s].to(device)
            if s == 0 and live_frame.poses is None:
                live_frame.poses = (
                    torch.eye(4, dtype=torch.float, device=device)
                    .view(1, 1, 4, 4)
                    .repeat(batch_size, 1, 1, 1)
                )
            pointclouds, live_frame.poses = self.step(
                pointclouds, live_frame, prev_frame)
            prev_frame = None if self.gt_odom else live_frame
            recovered_poses[:, s] = live_frame.poses[:, 0]
        return pointclouds, recovered_poses

    def step(
        self,
        pointclouds: Pointclouds,
        live_frame: RGBDImages,
        prev_frame: Optional[RGBDImages] = None,
    ):
        r"""Updates global map pointclouds with a SLAM step on `live_frame`.
        If `prev_frame` is not None, computes the relative transformation between `live_frame`
        and `prev_frame` using the selected odometry provider. If `prev_frame` is None,
        use the pose from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages or None): Input batch of previous frames (at time step :math:`t-1`).
                Must have sequence length of 1. If None, will (skip calling odometry provider and) use the pose
                from `live_frame`. Default: None
            

        Returns:
            tuple: tuple containing:

            - pointclouds (gradslam.Pointclouds): Updated :math:`B` global maps
            - poses (torch.Tensor): Poses for the live_frame batch

        Shape:
            - poses: :math:`(B, 1, 4, 4)`
        """
        live_frame.poses = self.odom.localize(pointclouds, live_frame, prev_frame)
        import pdb; pdb.set_trace()
        pointclouds = self.map.update_map(pointclouds, live_frame)
        return pointclouds, live_frame.poses


