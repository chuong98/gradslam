import torch

from gradslam.core.structures import Pointclouds
from .pcl_base import PCLOdometry
from .pcl_odom_utils import point_to_plane_ICP
from gradslam.models.builder import ODOMETRY

@ODOMETRY.register_module()
class ICPOdometry(PCLOdometry):
    r"""ICP odometry provider using a point-to-plane error metric. Computes the relative transformation between
    a pair of `gradslam.Pointclouds` objects using ICP (Iterative Closest Point). Uses LM (Levenberg-Marquardt) solver.
    """

    
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
        initial_transform = super().provide(maps_pointclouds, frames_pointclouds)
        transforms = []
        for b in range(len(maps_pointclouds)):
            transform, _ = point_to_plane_ICP(
                frames_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.normals_list[b].unsqueeze(0),
                initial_transform,
                numiters=self.numiters,
                damp=self.damp,
                dist_thresh=self.dist_thresh,
            )

            transforms.append(transform)

        return torch.stack(transforms).unsqueeze(1)

