import os
import warnings
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from ..geometry.geometryutils import relative_transformation
from torch.utils import data

from . import datautils

__all__ = ["Realsense"]


class Realsense(data.Dataset):
    """
    A torch Dataset for loading in the Realsense dataset.
    Structure of the Realsense dataset:
       
        ├── Realsense
        │   ├── depth/
        │   ├── rgb/
        │   ├── associations.txt
        │   └── intrinsics.txt

    Examples::

        >>> dataset = Realsense("path-to-Realsense-dataset", seqlen=20, width=320, height=240)
        >>> loader = DataLoader(dataset=dataset, batch_size=2)
        >>> colors, depths, intrinsics = next(iter(loader))

    """

    def __init__(
        self,
        basedir: str,
        seqlen: int = 4,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        height: int = 480,
        width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
    ):
        super(Realsense, self).__init__()

        basedir = os.path.normpath(basedir)
        self.seqlen = seqlen
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        dilation = dilation if dilation is not None else 0
        self.dilation = dilation
        stride = stride if stride is not None else seqlen * (dilation + 1)
        self.stride = stride
        start = start if start is not None else 0
        self.start = start
        self.end = end
        
        association_file = os.path.join(basedir, 'associations.txt')

        # Get a list of all color and depth files
        data_colorfiles, data_depthfiles = [], []
        with open(association_file, 'r') as f:
            lines = f.readlines()
        if end is None:
            end = len(lines)
            self.end = end
        lines = lines[start:end]
        
        for line in lines:
            line = line.split()
            data_depthfiles.append(os.path.normpath(os.path.join(basedir, line[1])))
            data_colorfiles.append(os.path.normpath(os.path.join(basedir, line[3])))

        # Assign color and depth files to each sequence
        data_len = len(data_depthfiles)
        idx = np.arange(seqlen) * (dilation + 1)
        colorfiles, depthfiles = [], []
        for start_idx in range(0, data_len, stride):
            if (start_idx + idx[-1] >= data_len):
                break

            idxs = start_idx + idx
            colorfiles.append([data_colorfiles[i] for i in idxs])
            depthfiles.append([data_depthfiles[i] for i in idxs])

        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles
        self.depthfiles = depthfiles

        # Camera intrinsics matrix
        fx, fy, ppx, ppy  = open(os.path.join(basedir, 'intrinsics.txt')).read().split()
        self.intrinsics = torch.tensor([[float(fx), 0, float(ppx), 0], [0, -float(fy), float(ppy), 0], [0, 0, 1, 0], [0, 0, 0, 1]]).unsqueeze(0)

        # Scaling factor for depth images
        self.scaling_factor = 5000.0

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

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
            color, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
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
            depth = datautils.channels_first(depth)
        return depth / self.scaling_factor

    def __getitem__(self, idx: int):
        r"""Returns the data from the sequence at index idx.

        Returns:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequence of depths of each frame
            pose_seq (torch.Tensor): Sequence of poses of each frame
            intrinsics (torch.Tensor): Intrinsics for the current sequence

        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Read in the color, depth and intrinstics info.
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]

        color_seq, depth_seq = [], []
        for i in range(self.seqlen):
            color = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
            color = self._preprocess_color(color)
            color = torch.from_numpy(color)
            color_seq.append(color)

            depth = np.asarray(imageio.imread(depth_seq_path[i]), dtype=np.int64)
            depth = self._preprocess_depth(depth)
            depth = torch.from_numpy(depth)
            depth_seq.append(depth)

        output = []
        color_seq = torch.stack(color_seq, 0).float()
        output.append(color_seq)
        
        depth_seq = torch.stack(depth_seq, 0).float()
        output.append(depth_seq)

        output.append(self.intrinsics)

        return tuple(output)
