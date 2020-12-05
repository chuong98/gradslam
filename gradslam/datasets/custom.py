import os
import warnings
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from gradslam.core.geometry.geometryutils import relative_transformation
from torch.utils.data import Dataset

from gradslam.datasets.pipelines import data_utils
from .builder import DATASETS
import mmcv
from mmcv.parallel import DataContainer as DC
from .pipelines.formating import to_tensor

@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(
        self,
        seqlen: int = 4,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        collect_keys=['img','depth','intrinsics','pose','transform'],
    ):
        super().__init__()
        dilation = dilation if dilation is not None else 0
        stride = stride if stride is not None else seqlen * (dilation + 1)
        start = start if start is not None else 0

        assert isinstance(seqlen,int) and seqlen > 0, f"seqlen must be positive integer. Got {seqlen}."
        assert isinstance(dilation, int) and dilation >= 0, f'dilation must be non-negative integer. Got {dilation}.'
        assert isinstance(stride,int) and stride > 0, f"stride must be positive. Got {stride}."
        assert isinstance(start, int) and start >=0, f"start must be positive. Got {start}."
        assert (isinstance(end,int) and end > start) or end is None, f"end ({end}) must be None or greater than start ({start})"

        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation
        self.start = start
        self.end = end

        self.collect_keys = collect_keys
        
        # load annotations 
        self.load_annotations()


    def load_annotations(self):
        """Load annotation from annotation file."""
        raise NotImplementedError

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

    def __getitem__(self, idx: int):
        r"""Returns the data from the sequence at index idx.

        Returns:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequence of depths of each frame
            pose_seq (torch.Tensor): Sequence of poses of each frame
            transform_seq (torch.Tensor): Sequence of transformations between each frame in the sequence and the
                previous frame. Transformations are w.r.t. the first frame in the sequence having identity pose
                (relative transformations with first frame's pose as the reference transformation). First
                transformation in the sequence will always be `torch.eye(4)`.
            intrinsics (torch.Tensor): Intrinsics for the current sequence
            framename (str): Name of the frame

        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - transform_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Get filename and img metas
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]
        framename = self.framenames[idx]
        timestamp_seq = self.timestamps[idx]
        img_metas=dict(framename=framename,timestamp_seq=timestamp_seq,
                        height=self.height,width=self.width)
        results = dict(img_metas=img_metas)
        poses=None
        if self.load_pose:
            pose_pointquat_seq = self.poses[idx]
            poses = self._homogenPoses(pose_pointquat_seq)

        return self.pipeline(results, color_seq_path, depth_seq_path, poses)

    def pipeline(self, results, color_seq_path, depth_seq_path, poses):
        for key in self.collect_keys:
            if key=='img':
                color_seq= []
                for i in range(self.seqlen):
                    color = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
                    color = self._preprocess_color(color)
                    color = to_tensor(color)
                    color_seq.append(color)
                color_seq = torch.stack(color_seq, 0).float()
                results['color_seq'] = DC(color_seq,stack=True)

            elif key=='depth':
                depth_seq =[]
                for i in range(self.seqlen):
                    depth = np.asarray(imageio.imread(depth_seq_path[i]), dtype=np.int64)
                    depth = self._preprocess_depth(depth)
                    depth = to_tensor(depth)
                    depth_seq.append(depth)
                depth_seq = torch.stack(results['depth_seq'], 0).float()
                results['depth_seq'] = DC(depth_seq,stack=True)

            elif key=='intrinsics':
                results['intrinsics']=to_tensor(self.intrinsics)

            elif key=='pose':
                pose_seq = [to_tensor(pose) for pose in poses]
                pose_seq = torch.stack(pose_seq, 0).float()
                pose_seq = self._preprocess_poses(pose_seq)
                results['pose_seq']=DC(pose_seq)

            elif key=='transform':
                transform_seq = data_utils.poses_to_transforms(poses)
                transform_seq = [to_tensor(x).float() for x in transform_seq]
                transform_seq = torch.stack(transform_seq, 0).float()
                results['transform_seq']=transform_seq

        return results

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.normalize_color:
            color = data_utils.normalize_image(color)
        if self.channels_first:
            color = data_utils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = data_utils.channels_first(depth)
        return depth / self.scaling_factor

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
    
    def _homogenPoses(self, poses_point_quaternion):
        r"""Converts a list of 3D point unit quaternion poses to a list of homogeneous poses

        Args:
            poses_point_quaternion (list of np.ndarray): List of np.ndarray 3D point unit quaternion
                poses, each of shape :math:`(7,)`.

        Returns:
            list of np.ndarray: List of homogeneous poses in np.ndarray format. Each np.ndarray
                has a shape of :math:`(4, 4)`.
        """
        return [
            data_utils.pointquaternion_to_homogeneous(pose)
            for pose in poses_point_quaternion
        ]