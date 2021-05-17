from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from tqdm import tqdm
from detectors.car_pose import CarPoseDetector
from opts import Opts
import shutil

from utils.utils import AverageMeter

# time_stats = ['net', 'dec']
def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.heads = {'hm': 3, 'hps': 20, 'rot': 8, 'dim': 3, 'prob': 1}
    # opt.heads = {'hm': 3, 'hps': 20, 'rot': 8, 'dim': 3, 'prob': 1, 'reg': 2, 'wh': 2}

    detector = CarPoseDetector(opt)
    if os.path.exists(opt.results_dir):
        shutil.rmtree(opt.results_dir,True)

    with open(opt.demo, 'r') as f:
        lines = f.readlines()
    image_names = [os.path.join(opt.data_dir + '/kitti/image/', img[:6] + '.png') for img in lines]
    depth_names = [os.path.join(opt.data_dir + '/kitti/depth_adabin/', img[:6] + '.png') for img in lines]

    time_tol=0
    num=0

    time_meter = AverageMeter()

    for i in tqdm(range(0, len(image_names))):
        image_name = image_names[i]
        depth_name = depth_names[i]
        num+=1
        ret, engine_time = detector.run(image_name, depth_name)
        if i > 40:
            time_meter.update(engine_time)
            print("Average engine time: ", time_meter.avg)

    # # NOTE Fulfil empty files
    # for (image_name) in image_names:
    #     image_id = image_name.split('/')[-1].split('.')[0]
    #     pred_filename = opt.results_dir + '/' 'data' + '/' + image_id + '.txt'
    #     if not os.path.isfile(pred_filename):
    #         print('NOT EXIST', pred_filename)
    #         with open(pred_filename, 'a') as fp:
    #             pass

if __name__ == '__main__':
    opt = Opts().init()
    main(opt)
