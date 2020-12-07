from .base import BaseSLAM
from ..builder import SLAM

@SLAM.register_module()
class ICPSLAM(BaseSLAM):
    r"""ICP-SLAM for batched sequences of RGB-D images.

    Args:
        odom_cfg (dict): Odometry method to be used from {'gt', 'icp', 'gradicp'}. Default: 'gradicp'
            dsratio (int): Downsampling ratio to apply to input frames before ICP. Only used if `odom` is
                'icp' or 'gradicp'. Default: 4
            numiters (int): Number of iterations to run the optimization for. Only used if `odom` is
                'icp' or 'gradicp'. Default: 20
            damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Only used if `odom` is
                'icp' or 'gradicp'. Default: 1e-8
            dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                    Only used if `odom` is 'icp' or 'gradicp'. Default: None
            lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be
                :math:`\frac{1}{\text{lambda_max}}`). Only used if `odom` is 'gradicp'.
            B (float or int): gradLM falloff control parameter (see GradICPOdometry description).
                Only used if `odom` is 'gradicp'.
            B2 (float or int): gradLM control parameter (see GradICPOdometry description).
                Only used if `odom` is 'gradicp'.
            nu (float or int): gradLM control parameter (see GradICPOdometry description).
                Only used if `odom` is 'gradicp'.
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            the CPU. Default: None
    """
    # TODO: Try to have nn.Module features supported
    def __init__(
        self,
        odom_cfg=dict(type="GradICP",dsratio = 4, numiters = 20, damp = 1e-8, 
                        dist_thr = None, lambda_max = 2.0, 
                        B = 1.0, B2 = 1.0, nu = 200.0),
        map_cfg=dict(type='AggregateMap', inplace=True),
        *args,**kwargs
    ):
        super().__init__(odom_cfg,map_cfg, *args,**kwargs)
        