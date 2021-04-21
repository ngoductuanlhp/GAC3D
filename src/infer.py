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
    opt.heads = {'hm': 1, 'hps': 20, 'rot': 8, 'dim': 3}
    # opt.heads = {'hm': 1, 'hps': 20, 'rot': 8, 'dim': 3, 'prob': 1, 'reg': 2, 'wh': 2}
    # opt.hm_hp=False
    # opt.reg_offset=False
    # opt.reg_hp_offset=False
    detector = CarPoseDetector(opt)
    if os.path.exists(opt.results_dir):
        shutil.rmtree(opt.results_dir, True)

    with open(opt.demo, 'r') as f:
        lines = f.readlines()
    image_names = [os.path.join(
        opt.data_dir + '/kitti/image/', img[:6] + '.png') for img in lines]
    depth_names = [os.path.join(
        opt.data_dir + '/kitti/depth_adabin/', img[:6] + '.png') for img in lines]

    time_tol = 0
    num = 0

    for i in tqdm(range(0, len(image_names))):
        image_name = image_names[i]
        depth_name = depth_names[i]
        num += 1
        ret = detector.run(image_name, depth_name)
        # time_str = ''
        # for stat in time_stats:
        #     time_tol=time_tol+ret[stat]
        #     time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        # time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
        # print(time_str)

    # NOTE Fulfil empty files
    for (image_name) in image_names:
        image_id = image_name.split('/')[-1].split('.')[0]
        pred_filename = opt.results_dir + '/' 'data' + '/' + image_id + '.txt'
        if not os.path.isfile(pred_filename):
            print('NOT EXIST', pred_filename)
            with open(pred_filename, 'a') as fp:
                pass


if __name__ == '__main__':
    opt = Opts().init()
    main(opt)
