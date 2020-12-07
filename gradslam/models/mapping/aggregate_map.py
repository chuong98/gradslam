from gradslam.core.structures.utils import pointclouds_from_rgbdimages
from gradslam.core.structures import Pointclouds, RGBDImages
from .base import BaseMap
from ..builder import MAP

@MAP.register_module()
class AggregateMap(BaseMap):
    """ Args:
        inplace (bool): Can optionally update the pointclouds and live_frame poses in-place. Default: False
    """
    def __init__(self, inplace=True):
        self.inplace=inplace

    def update_map(self, 
        pointclouds: Pointclouds, 
        live_frame: RGBDImages, 
    ) -> Pointclouds:
        r"""Aggregate points from live frames with global maps by appending the live frame points.

        Args:
            pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
                (ccounts).
            live_frame (gradslam.RGBDImages): Live frames from the latest sequence
            inplace (bool): Can optionally update the pointclouds in-place. Default: False

        Returns:
            gradslam.Pointclouds: Updated Pointclouds object containing global maps.

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
        new_pointclouds = pointclouds_from_rgbdimages(live_frame, global_coordinates=True)
        if not self.inplace:
            pointclouds = pointclouds.clone()
        pointclouds.append_points(new_pointclouds)
        return pointclouds