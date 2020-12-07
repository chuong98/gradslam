from .base import BaseSLAM
from ..builder import SLAM

@SLAM.register_module()
class PointFusionSLAM(BaseSLAM):
    r"""Point-based Fusion (PointFusion for short) SLAM for batched sequences of RGB-D images
    (See Point-based Fusion `paper <http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf>`__).

    Args:
        odom_cfg (dict): Odometry method to be used from {'gt', 'icp', 'gradicp'}. Default: 'GradICPOdometry', with following params:
            dsratio (int): Downsampling ratio to apply to input frames before ICP. Only used if `odom` is
                'icp' or 'gradicp'. Default: 4
            numiters (int): Number of iterations to run the optimization for. Only used if `odom` is
                'icp' or 'gradicp'. Default: 20
            damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Only used if `odom` is
                'icp' or 'gradicp'. Default: 1e-8
            dist_thr (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                    Only used if `odom` is 'icp' or 'gradicp'. Default: None
            lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be
                :math:`\frac{1}{\text{lambda_max}}`). Only used if `odom` is 'gradicp'.
            B (float or int): gradLM falloff control parameter (see GradICPOdometry description).
                Only used if `odom` is 'gradicp'.
            B2 (float or int): gradLM control parameter (see GradICPOdometry description).
                Only used if `odom` is 'gradicp'.
            nu (float or int): gradLM control parameter (see GradICPOdometry description).
                Only used if `odom` is 'gradicp'.

        map_cfg (dict): Mapping Fusion methods. Default 'PointFusionMap', with following params:
            dist_thr (float or int): Distance threshold.
            angle_thr (float or int): Angle threshold
            dot_thr (float or int): Dot product threshold (cos(angle_thr)).
            sigma (torch.Tensor or float or int): Width of the gaussian bell. Original paper uses 0.6 emperically.
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            the CPU. Default: None


    Examples::

    >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
    >>> slam = PointFusionSLAM(odom_cfg=dict(type='gt'))
    >>> pointclouds, poses = slam(rgbdimages)
    >>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])
    """

    def __init__(
        self,
        odom_cfg=dict(type="GradICPOdometry",dsratio = 4, numiters = 20, damp = 1e-8, 
                        dist_thr = None, lambda_max = 2.0, 
                        B = 1.0, B2 = 1.0, nu = 200.0),
        map_cfg=dict(type="PointFusionMap", dist_thr = 0.05, angle_thr= 20, sigma=0.6, inplace=True),
        *args,**kwargs
    ):
        super().__init__(odom_cfg,map_cfg,*args,**kwargs)
         
