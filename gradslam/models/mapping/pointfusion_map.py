from typing import Union
import warnings
import math
import torch

from gradslam.core.structures import Pointclouds, RGBDImages
from .base import BaseMap
from ..builder import MAP
from .mapping_utils import find_correspondences, fuse_with_map

@MAP.register_module()
class PointFusionMap(BaseMap):
    def __init__(self,         
        dist_thr: Union[float, int],
        angle_thr: Union[float,int],
        dot_thr: Union[float, int],
        sigma: Union[float, int],
        inplace: bool = False):
        r"""
            dist_thr (float or int): Distance threshold.
            angle_thr (float or int): Angle Threshold
            dot_thr (float or int): Dot product threshold (cos(angle_thr)).
            sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
            inplace (bool): Can optionally update the pointclouds in-place. Default: False
        """
        self.dist_thr = dist_thr
        if angle_thr is not None:
            assert dot_thr is None, "only one of `angle_thr` or `dot_thr` can be set."
            if not ((0 <= angle_thr) and (angle_thr <= 90)):
                warnings.warn(
                    "Angle threshold ({}) should be non-negative and <=90.".format(angle_thr)
                )
            rad_th = (angle_thr * math.pi) / 180
            self.angle_thr = angle_thr
            self.dot_thr = torch.cos(rad_th) if torch.is_tensor(rad_th) else math.cos(rad_th)
        else:
            assert dot_thr is not None, "either angle_thr or dot_thr must be set"
            self.dot_thr = dot_thr
            angle_thr = torch.arccos(dot_thr) if torch.is_tensor(dot_thr) else math.arcos(dot_thr)
            self.angle_thr = 180/math.pi *angle_thr
        self.sigma = sigma
        self.inplace = inplace

    def update_map(self,
        pointclouds: Pointclouds,
        rgbdimages: RGBDImages,
    ) -> Pointclouds:
        r"""Updates pointclouds in-place given the live frame RGB-D images using PointFusion.
        (See Point-based Fusion `paper <http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf>`__).

        Args:
            pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
                (ccounts).
            rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        Returns:
            gradslam.Pointclouds: Updated Pointclouds object containing global maps.

        """
        pc2im_bnhw = find_correspondences(pointclouds, rgbdimages, self.dist_thr, self.dot_thr)
        pointclouds = fuse_with_map(pointclouds, rgbdimages, pc2im_bnhw, self.sigma, self.inplace)

        return pointclouds
