from .data_utils import *
import os.path as osp

import cv2
import numpy as np
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from .formating import to_tensor
from ..builder import PIPELINES
from gradslam.core.geometry.geometryutils import relative_transformation

@PIPELINES.register_module()
class ColorSeqFormatBundle(object):
    def __init__(self, 
                normalize_color=True,
                channels_first=True,
                to_float32=True,
                file_client_args=dict(backend='disk') ):
        self.normalize_color = normalize_color
        self.channels_first = channels_first
        self.to_float32=to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
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
        color_seq=[]
        resize=results.get('resize',None)
        for color_fn in results['color_seq_path']:
            if results['img_prefix'] is not None:
                filename = osp.join(results['img_prefix'],color_fn)
            else:
                filename = color_fn
            color = self._load_img(filename)
            color = self._preprocess_color(color,resize)
            color = to_tensor(color)
            color_seq.append(color)
        color_seq = torch.stack(color_seq, 0).float()
        results['color_seq'] = DC(color_seq,stack=True)
        return results

    def _load_img(self,filename):
        # img = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag='color')
        if self.to_float32:
            img = img.astype(np.float32)
        return img 

    def _preprocess_color(self, color: np.ndarray, resize):
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
        if resize:
            color = cv2.resize(color, resize, 
                interpolation=cv2.INTER_LINEAR)
        if self.normalize_color:
            color = normalize_image(color)
        if self.channels_first:
            color = channels_first(color)
        return color

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class DepthSeqFormatBundle(object):
    def __init__(self, 
                scaling_factor=1, 
                channels_first=True,
                to_float32=True,
                file_client_args=dict(backend='disk')):
        self.scaling_factor = scaling_factor
        self.channels_first = channels_first
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self,results):
        depth_seq =[]
        resize=results.get('resize',None)
        for depth_fn in results['depth_seq_path']:
            if results['depth_prefix'] is not None:
                filename = osp.join(results['depth_prefix'],depth_fn)
            else:
                filename = depth_fn
            depth = self._load_img(filename)
            depth = self._preprocess_depth(depth,resize)
            depth = to_tensor(depth)
            depth_seq.append(depth)
        depth_seq = torch.stack(depth_seq, 0)
        results['depth_seq'] = DC(depth_seq,stack=True)
        return results

    def _load_img(self,filename):
        # img = np.asarray(imageio.imread(depth_fn), dtype=np.int64)
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag='unchanged')
        if self.to_float32:
            img = img.astype(np.float32)
        return img
        
    def _preprocess_depth(self, depth: np.ndarray, resize):
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
        if resize:
            depth = cv2.resize(
                depth.astype(float),
                resize,
                interpolation=cv2.INTER_NEAREST,
            )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = channels_first(depth)
        
        return depth / self.scaling_factor

    
@PIPELINES.register_module()
class PoseSeqFormatBundle(object):         
    def __call__(self, results):
        pose_seq = [to_tensor(pose) for pose in results['poses']]
        pose_seq = torch.stack(pose_seq, 0).float()
        pose_seq = self._preprocess_poses(pose_seq)
        results['pose_seq']=DC(pose_seq)
        return results

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

@PIPELINES.register_module()
class HomoTransformSeqFormatBundle(object):  
    def __call__(self,results):
        transform_seq = poses_to_transforms(results['poses'])
        transform_seq = [to_tensor(x).float() for x in transform_seq]
        transform_seq = torch.stack(transform_seq, 0).float()
        results['transform_seq']=transform_seq
        return results
