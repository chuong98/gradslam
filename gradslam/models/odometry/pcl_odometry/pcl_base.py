from typing import Union

import torch
from kornia.geometry.linalg import compose_transformations

from gradslam.core.structures import Pointclouds, RGBDImages
from .pcl_odom_utils import downsample_pointclouds, downsample_rgbdimages
from gradslam.models.mapping.mapping_utils import find_active_map_points

from ..base import Odometry

__all__ = ["PCLOdometry"]


class PCLOdometry(Odometry):
    r"""PCL odometry provider take point cloud inputs to compute camera transformation.
    """

    def __init__(
        self,
        numiters: int = 20,
        damp: float = 1e-8,
        dist_thr: Union[float, int, None] = None,
        dsratio: int = 4,
    ):
        r"""Initializes internal ICPOdometry state.

        Args:
            numiters (int): Number of iterations to run the optimization for. Default: 20
            damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Default: 1e-8
            dist_thr (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                Default: None

        """
        self.numiters = numiters
        self.damp = damp
        self.dist_thr = dist_thr
        self.dsratio = dsratio
        
    def provide(
        self,
        maps_pointclouds: Pointclouds,
        frames_pointclouds: Pointclouds,
    ) -> torch.Tensor:
        r"""Uses ICP to compute the relative homogenous transformation that, when applied to `frames_pointclouds`,
        would cause the points to align with points of `maps_pointclouds`.

        Args:
            maps_pointclouds (gradslam.Pointclouds): Object containing batch of map pointclouds of batch size
                :math:`(B)`
            frames_pointclouds (gradslam.Pointclouds): Object containing batch of live frame pointclouds of batch size
                :math:`(B)`

        Returns:
            torch.Tensor: The relative transformation that would align `maps_pointclouds` with `frames_pointclouds`

        Shape:
            - Output: :math:`(B, 1, 4, 4)`

        """
        if not isinstance(maps_pointclouds, Pointclouds):
            raise TypeError(
                "Expected maps_pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(maps_pointclouds)
                )
            )
        if not isinstance(frames_pointclouds, Pointclouds):
            raise TypeError(
                "Expected frames_pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(frames_pointclouds)
                )
            )
        if maps_pointclouds.normals_list is None:
            raise ValueError(
                "maps_pointclouds missing normals. Map normals must be provided if using ICPOdometry"
            )
        if len(maps_pointclouds) != len(frames_pointclouds):
            raise ValueError(
                "Batch size of maps_pointclouds and frames_pointclouds should be equal ({0} != {1})".format(
                    len(maps_pointclouds), len(frames_pointclouds)
                )
            )

        device = maps_pointclouds.device
        initial_transform = torch.eye(4, device=device)
        return initial_transform


    def localize(
        self, 
        pointclouds: Pointclouds, 
        live_frame: RGBDImages, 
        prev_frame: RGBDImages
    ):
        r"""Compute the poses for `live_frame`. If `prev_frame` is not None, computes the relative
        transformation between `live_frame` and `prev_frame` using the selected odometry provider.
        If `prev_frame` is None, use the pose from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages or None): Input batch of previous frames (at time step :math:`t-1`).
                Must have sequence length of 1. If None, will (skip calling odometry provider and) use the pose
                from `live_frame`. Default: None

        Returns:
            torch.Tensor: Poses for the live_frame batch

        Shape:
            - Output: :math:`(B, 1, 4, 4)`
        """
        if not isinstance(pointclouds, Pointclouds):
            raise TypeError(
                "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(pointclouds)
                )
            )
        if not isinstance(live_frame, RGBDImages):
            raise TypeError(
                "Expected live_frame to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(live_frame)
                )
            )
        if not isinstance(prev_frame, (RGBDImages, type(None))):
            raise TypeError(
                "Expected prev_frame to be of type gradslam.RGBDImages or None. Got {0}.".format(
                    type(prev_frame)
                )
            )
        
        if prev_frame is not None and not prev_frame.has_poses:
            raise ValueError("`prev_frame` should have poses, but did not.")

        if prev_frame is None:
            if not live_frame.has_poses:
                raise ValueError("`live_frame` must have poses when `prev_frame` is None")
            return live_frame.poses

        live_frame.poses = prev_frame.poses
        frames_pc = downsample_rgbdimages(live_frame, self.dsratio)
        pc2im_bnhw = find_active_map_points(pointclouds, prev_frame)
        maps_pc = downsample_pointclouds(pointclouds, pc2im_bnhw, self.dsratio)
        transform = self.provide(maps_pc, frames_pc)

        return compose_transformations(
            transform.squeeze(1), prev_frame.poses.squeeze(1)
        ).unsqueeze(1)
