from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from tqdm import tqdm
from detectors.car_pose import CarPoseDetector
from opts import Opts
import shutil

# time_stats = ['net', 'dec']
def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.heads = {'hm': 3, 'hps': 20, 'rot': 8, 'dim': 3, 'prob': 1}
    # opt.heads = {'hm': 3, 'hps': 20, 'rot': 8, 'dim': 3, 'prob': 1, 'reg': 2, 'wh': 2}

    detector = CarPoseDetector(opt, onnx=True)

    image_name = os.path.join(opt.data_dir + '/kitti/image/', '000001.png')
    depth_name = os.path.join(opt.data_dir + '/kitti/depth_adabin/', '000001.png')
    ret = detector.run(image_name, depth_name)


if __name__ == '__main__':
    opt = Opts().init()
    main(opt)
