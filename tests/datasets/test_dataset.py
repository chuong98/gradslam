import argparse
import os

import mmcv
import numpy as np
import torch

from gradslam.utils import PublicConfig as Config
from gradslam.datasets import build_dataset

def build_index_html(file_names):
    images = ""
    for file_name in file_names:
        images +=  f'<img src="{file_name}" alt="Trulli" width="500" height="333">'
    s=f"""
    <!DOCTYPE html>
<html>
  <body>

    <h2>HTML Image</h2>
    {images}
    </body>
    </html>>
    """
    return s

def parse_args():
    parser = argparse.ArgumentParser(
        description='CCSLAM test (and eval) a dataset')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--num_imgs', type=int, default=10)
    parser.add_argument('--out-dir', default="./cache")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)


    imgs    = []
    names   = []

    for idx in np.random.choice(len(dataset), args.num_imgs):
        item = dataset.__getitem__(idx)
        import pdb; pdb.set_trace()
    #     imgs.append(item['img'].data)
    # imgs = mmcv.tensor2imgs(torch.stack(imgs), mean, std, to_rgb=True)

    # for img,bbox in zip(imgs,bboxes):
    #     mmcv.visualization.imshow_bboxes(img, bbox, show=False)

    # for i, img in enumerate(imgs):
    #     name =  f'{i}.jpg'
    #     out_path = os.path.join(args.out_dir,name)
    #     print(out_path)
    #     names.append(name)
    #     mmcv.imwrite(img,out_path)

    # with open(os.path.join(args.out_dir, 'index.html'), "w") as f:
    #     f.write(build_index_html(names))
    #     print(os.path.join(args.out_dir, 'index.html'))