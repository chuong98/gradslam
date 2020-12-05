import warnings
from .custom import CustomDataset
import os.path as osp

from .builder import DATASETS
@DATASETS.register_module()
class ICLDataset(CustomDataset):
    """
        example usage:
        data_root='/data/ICL/living_room_traj1_frei_png/'
        ICLDataset(
            ann_file=data_root + 'associations.txt',
            data_root= data_root,
            img_prefix='',
            depth_prefix='',
        )
    """
    def __init__(self, gt_pose, *args, **kwargs):
        self.gt_pose=gt_pose 
        super().__init__(*args,**kwargs)

    def load_annotations(self, ann_file):
        assert osp.basename(ann_file)=='associations.txt'
        with open(ann_file, "r") as f:
            lines = f.readlines()
            if self.end is None:
                self.end = len(lines)
            if self.end > len(lines):
                msg = "end was larger than number of frames in trajectory: {0} > {1}"
                warnings.warn(msg.format(self.end, len(lines)))
            lines = lines[self.start:self.end]
        
        if self.gt_pose:
            
        data_infos=[]
        for line in lines:
            # associate.txt has the following format
            # 0 depth/0.png 0 rgb/0.png
            # 1 depth/1.png 1 rgb/1.png
            line = line.strip().split()
            data_infos.append(dict(id=line[0], filename=line[3], depth_filename=[1]))



        