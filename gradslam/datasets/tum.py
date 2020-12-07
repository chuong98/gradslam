import warnings
import numpy as np
from typing import Optional, Union
from .custom import CustomDataset
from .builder import DATASETS
from . import tumutils
from .pipelines.data_utils import pointquaternion_to_homogeneous

@DATASETS.register_module()
class TUMDataset(CustomDataset):
    """
        example usage:
        data_root='/data/TUM/rgbd_dataset_freiburgX_NAME/'
        TUMDataset(
            data_root= data_root,
            ann_files=dict(rgb='rgb.txt',depth='depth.txt',gt_pose='groundtruth.txt'),
            img_prefix='',
            depth_prefix='',
        )

        TUMM dataset is downloaded from `here <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>`__.
        Expects similar to the following folder structure for the TUM dataset:

        .. code-block::


        | ├── TUM
        | │   ├── rgbd_dataset_freiburg1_rpy
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── accelerometer.txt
        | │   │   ├── depth.txt
        | │   │   ├── groundtruth.txt
        | │   │   └── rgb.txt
        | │   ├── rgbd_dataset_freiburg1_xyz
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── accelerometer.txt
        | │   │   ├── depth.txt
        | │   │   ├── groundtruth.txt
        | │   │   └── rgb.txt
        | │   ├── ...
    """

    def __init__(self,
                *args,**kwargs):
        intrinsic_cfg=dict(fx=525.0,fy=525.0,cx=319.5,cy=239.5,H=480,W=640)
        super().__init__(intrinsics_cfg=intrinsic_cfg,
                        *args,**kwargs)  
                             
    def load_annotations(self):
        super().load_annotations()
        rgb_text_file = self.ann_files['rgb']
        depth_text_file = self.ann_files['depth']
        self.pose_file = self.ann_files.get('gt_pose',None)

        # Get a list of all color, depth, pose, label and intrinsics files.
        seq_colorfiles, seq_depthfiles = [], []
        seq_poses, seq_framenames = [], []
        associations, seq_timestamps = self._findAssociations(
                rgb_text_file, depth_text_file, self.pose_file
            )

        for association in associations:
            seq_colorfiles.append(association[0])
            seq_depthfiles.append(association[1])
            if self.pose_file:
                seq_poses.append(association[2])
            seq_framenames.append(association[0][3:-4])

        # Store the files to class members
        self.timestamps = []
        self.poses = []
        idx = np.arange(self.seqlen) * (self.dilation + 1)
        num_frames = len(seq_colorfiles)
        for start_ind in range(0, num_frames, self.stride):
            if (start_ind + idx[-1]) >= num_frames:
                break
            inds = start_ind + idx
            self.colorfiles.append([seq_colorfiles[i] for i in inds])
            self.depthfiles.append([seq_depthfiles[i] for i in inds])
            self.framenames.append(", ".join([seq_framenames[i] for i in inds]))
            self.timestamps.append([seq_timestamps[i] for i in inds])
            if self.pose_file:
                self.poses.append([seq_poses[i] for i in inds])

        self.num_sequences = len(self.colorfiles)    

    def get_data_seq_info(self, idx):
        results= super().get_data_seq_info(idx)
        # Add pose
        if self.pose_file:
            pose_pointquat_seq = self.poses[idx]
            results['poses'] = self._homogenPoses(pose_pointquat_seq)
        # Add time stamp
        results['timestamps']=["rgb {} depth {} pose {}".format(*t) 
                                for t in self.timestamps[idx]]
        return results

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
            pointquaternion_to_homogeneous(pose)
            for pose in poses_point_quaternion
        ]

    def _findAssociations(
        self,
        rgb_text_file: str,
        depth_text_file: str,
        poses_text_file: Optional[str] = None,
        max_difference: float = 0.02,
    ):
        r"""Associates TUM color images, depth images and (optionally) poses based on un-synchronized time
        stamps and returns associations as tuples.

        Args:
            rgb_text_file (str): Path to "rgb.txt"
            depth_text_file (str): Path to "depth.txt"
            poses_text_file (str or None): Path to ground truth poses ("groundtruth.txt"). Default: None
            max_difference (float): Search radius for candidate generation. Default: 0.02

        Returns:
            associations (list of tuple): List of tuples, each tuple containing rgb frame path,
                depth frame path, and an np.ndarray for 3D point unit quaternion poses of shape :math:`(7,)`
                (rgb_frame_path, depth_frame_path, point_quaternion_npndarray).
            timestamps (list of tuple of str): Timestamps of matched rgb, depth and pose.
                The first dimension corresponds to the number of matches :math:`N`, and the second dimension
                stores the associated timestamps as (rgb_timestamp, depth_timestamp, pose_timestamp).

        """
        rgb_dict = tumutils.read_file_list(rgb_text_file, self.start, self.end)
        depth_dict = tumutils.read_file_list(depth_text_file)
        matches = tumutils.associate(rgb_dict, depth_dict, 0, float(max_difference))

        if poses_text_file is not None:
            poses_dict = tumutils.read_trajectory(poses_text_file, matrix=False)
            matches_dict = {match[1]: match[0] for match in matches}
            matches = tumutils.associate(
                matches_dict, poses_dict, 0, float(max_difference)
            )
            matches = [
                (matches_dict[match[0]], match[0], match[1]) for match in matches
            ]

        if poses_text_file is None:
            associations = [(rgb_dict[m[0]][0], depth_dict[m[1]][0]) for m in matches]
            timestamps = [(m[0], m[1], None) for m in matches]
        else:
            associations = [
                (
                    rgb_dict[m[0]][0],
                    depth_dict[m[1]][0],
                    np.array(poses_dict[m[2]], dtype=np.float32),
                )
                for m in matches
            ]
            timestamps = list(matches)
        return associations, timestamps