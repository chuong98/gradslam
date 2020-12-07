import os.path as osp
from numpy.lib.arraysetops import isin
import torch
from typing import Optional
from .pipelines.data_utils import scale_intrinsics
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose

@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(
        self,
        ann_files,
        pipeline,
        data_root=None,
        img_prefix='',
        depth_prefix='',
        seqlen: int = 4,
        intrinsics_cfg=dict(fx=1.0,fy=1.0,cx=1.0,cy=1.0,H=480,W=640),
        resize=None,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        super().__init__()
        assert isinstance(ann_files,str) or isinstance(ann_files,dict), \
            'ann_files must be a string path to a single file, or a dict of a set of path files' 
        self.ann_files= ann_files
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.depth_prefix = depth_prefix

        # join paths if data_root is specified
        if self.data_root is not None:
            def to_abs_path(file_path):
                return osp.join(self.data_root,file_path)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = to_abs_path(self.img_prefix)
            if not (self.depth_prefix is None or osp.isabs(self.depth_prefix)):
                self.depth_prefix = to_abs_path(self.depth_prefix)
            if isinstance(self.ann_files,str):
                if not osp.isabs(self.ann_files):
                    self.ann_files = to_abs_path(self.ann_files)
            else:
                for k,v in self.ann_files.items():
                    if not osp.isabs(v):
                        self.ann_files[k] = to_abs_path(v)

        # sequence data info
        dilation = 0 if dilation is None else dilation
        stride = seqlen * (dilation + 1) if stride is None else stride
        start = 0 if start is None else start

        assert isinstance(seqlen,int) and seqlen > 0, f"seqlen must be positive integer. Got {seqlen}."
        assert isinstance(dilation, int) and dilation >= 0, f'dilation must be non-negative integer. Got {dilation}.'
        assert isinstance(stride,int) and stride > 0, f"stride must be positive. Got {stride}."
        assert isinstance(start, int) and start >=0, f"start must be positive. Got {start}."
        assert (isinstance(end,int) and end > start) or end is None, f"end ({end}) must be None or greater than start ({start})"

        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation
        self.start = start
        self.end = end

        # load annotations 
        self.load_annotations()
        
        # Intrinsics matrix
        self.resize=resize
        self.intrinsics_cfg=intrinsics_cfg
        self.intrinsics = self.build_intrinsics(intrinsics_cfg,resize)

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def build_intrinsics(self,intrinsic_cfg, resize=None):
        intrinsics = torch.tensor(
            [[intrinsic_cfg['fx'],0,intrinsic_cfg['cx'],0], 
             [0, intrinsic_cfg['fy'],intrinsic_cfg['cy'],0], 
             [0, 0, 1, 0], 
             [0, 0, 0, 1]]
        ).float()
        if resize:
            height_downsample_ratio=resize[0]/intrinsic_cfg['H']
            width_downsample_ratio=resize[1]/intrinsic_cfg['W']
            intrinsics = scale_intrinsics(
                intrinsics, height_downsample_ratio, width_downsample_ratio
            ).unsqueeze(0)
        return intrinsics

    def load_annotations(self):
        """Load annotation from annotation file. Customize for each dataset"""
        self.num_sequences=0
        self.colorfiles = []
        self.depthfiles = []
        self.framenames = []
        self.timestamps = []

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

    def __getitem__(self, idx: int):
        r"""Returns the data from the sequence at index idx.

        Returns:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequence of depths of each frame
            pose_seq (torch.Tensor): Sequence of poses of each frame
            transform_seq (torch.Tensor): Sequence of transformations between each frame in the sequence and the
                previous frame. Transformations are w.r.t. the first frame in the sequence having identity pose
                (relative transformations with first frame's pose as the reference transformation). First
                transformation in the sequence will always be `torch.eye(4)`.
            intrinsics (torch.Tensor): Intrinsics for the current sequence
            framename (str): Name of the frame

        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - transform_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Get filename and img metas
        data_seq_info = self.get_data_seq_info(idx)
        results = self.pre_pipeline(data_seq_info)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['depth_prefix'] = self.depth_prefix
        results['resize']=self.resize
        results['intrinsics']=self.intrinsics
        return results

    def get_data_seq_info(self,idx):
        """Customize for each dataset """
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]
        framename = self.framenames[idx]
        return dict(color_seq_path=color_seq_path,
                    depth_seq_path=depth_seq_path,
                    framename=framename,
                    )

        

    
    