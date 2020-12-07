import warnings
import numpy as np

from .custom import CustomDataset
from .builder import DATASETS

@DATASETS.register_module()
class ICLDataset(CustomDataset):
    """
        example usage:
        data_root='/data/ICL/living_room_traj1_frei_png/'
        ICLDataset(
            data_root= data_root,
            ann_files=dict(associate_file='associations.txt',
                            gt_pose='livingRoom1n.gt.sim'),
            img_prefix='',
            depth_prefix='',
        )
    """
    def __init__(self, 
                *args, **kwargs):
        # preprocess trajectories to be a tuple or None
        intrinsic_cfg=dict(fx=481.2,fy=-480.0,cx=319.5,cy=239.5,H=480,W=640)
        super().__init__(intrinsics_cfg=intrinsic_cfg,
                        *args,**kwargs)

    def load_annotations(self):
        super().load_annotations()
        associate_file = self.ann_files['associate_file']
        self.pose_file = self.ann_files['gt_pose']
        traj_colorfiles, traj_depthfiles = [], []
        traj_poselinenums, traj_framenames = [], []
        if self.pose_file:
            self.posemetas=[]
        # Load file path for rgb and depth
        with open(associate_file, "r") as f:
            lines = f.readlines()
            if self.end is None:
                self.end = len(lines)
            if self.end > len(lines):
                msg = "end was larger than number of frames in trajectory: {0} > {1}"
                warnings.warn(msg.format(self.end, len(lines)))
            lines = lines[self.start:self.end]

        for line_num,line in enumerate(lines):
            # associate.txt has the following format
            # 0 depth/0.png 0 rgb/0.png
            # 1 depth/1.png 1 rgb/1.png
            line = line.strip().split()
            traj_colorfiles.append(line[3])
            traj_depthfiles.append(line[1])
            traj_poselinenums.append(line_num * 4)
            traj_framenames.append(line[0])

        traj_len = len(traj_colorfiles)
        idx = np.arange(self.seqlen) * (self.dilation + 1)
        for start_ind in range(0, traj_len, self.stride):
            if (start_ind + idx[-1]) >= traj_len:
                break
            inds = start_ind + idx
            self.colorfiles.append([traj_colorfiles[i] for i in inds])
            self.depthfiles.append([traj_depthfiles[i] for i in inds])
            self.framenames.append(", ".join([traj_framenames[i] for i in inds]))
            if self.pose_file:
                self.posemetas.append([traj_poselinenums[i] for i in inds])

        self.num_sequences = len(self.colorfiles)

    def get_data_seq_info(self, idx):
        results= super().get_data_seq_info(idx)
        if self.pose_file:
            results['poses'] = self._loadPoses(self.pose_file, self.posemetas[idx])
        return results

    def _loadPoses(self, pose_path, start_lines):
        r"""Loads poses from groundtruth pose text files and returns the poses
        as a list of numpy arrays.

        Args:
            pose_path (str): The path to groundtruth pose text file.
            start_lines (list of ints):

        Returns:
            poses (list of np.array): List of ground truth poses in
                    np.array format. Each np.array has a shape of [4, 4] if
                    homogen_coord is True, or a shape of [3, 4] otherwise.
        """
        pose = []
        poses = []
        parsing_pose = False
        with open(pose_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if not (i in start_lines or parsing_pose):
                continue
            parsing_pose = True
            line = line.strip().split()
            if len(line) != 4:
                msg = "Faulty poses file: Expected line {0} of the poses file {1} to contain pose matrix values, "
                msg += 'but it didn\'t. You can download "Global_RT_Trajectory_GT" from here:\n'
                msg += "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"
                raise ValueError(msg)
            pose.append(line)

            if len(pose) == 3:
                pose.append([0.0, 0.0, 0.0, 1.0])
                poses.append(np.array(pose, dtype=np.float32))
                pose = []
                parsing_pose = False

        return poses
        