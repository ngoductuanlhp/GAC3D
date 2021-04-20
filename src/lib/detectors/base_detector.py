from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
import os


def gen_ground(P2_prime, h, w, P0=None):
    baseline = 0.54
    relative_elevation = 1.65
    P2 = P2_prime

    fy = P2[1:2, 1:2]  # [B, 1, 1]
    cy = P2[1:2, 2:3]  # [B, 1, 1]
    Ty = P2[1:2, 3:4]  # [B, 1, 1]

    x_range = torch.arange(w)
    y_range = torch.arange(570)
    y_range = torch.clamp(y_range, int(cy) + 1)
    _, yy_grid = torch.meshgrid(x_range, y_range)
    yy_grid = torch.transpose(yy_grid, 0, 1)

    z = (fy * relative_elevation + Ty) / (yy_grid - cy + 1e-10)
    z = torch.clamp(z, 0, 75)
    z = z[:, 0]
    return z


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True
        self.image_path = ' '
        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
             [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0)
        self.const = self.const.to(self.opt.device)

        self.depth_mean = np.array(
            [30.83664619525601], np.float32).reshape(1, 1, 1)  # DISP
        self.depth_std = np.array(
            [19.992999492848206],  np.float32).reshape(1, 1, 1)

    def pre_process(self, images, depth):
        raise NotImplementedError

    def process(self, images, meta, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results, calib):
        raise NotImplementedError

    def read_clib(self, calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def read_clib0(self, calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 0:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def run(self, image_or_path_or_tensor, depth_name, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            self.image_path = image_or_path_or_tensor
            image = cv2.imread(image_or_path_or_tensor)
            depth = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)
            calib_path = os.path.join(
                self.opt.calib_dir, image_or_path_or_tensor[-10:-3]+'txt')
            calib_numpy = self.read_clib(calib_path)
            # calib_numpy0=self.read_clib0(calib_path)
            h, w = image.shape[0:2]
            ground_plane = gen_ground(
                calib_numpy, h, w).unsqueeze(0).to(self.opt.device)

            calib = torch.from_numpy(calib_numpy).unsqueeze(
                0).to(self.opt.device)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        scale_start_time = time.time()

        images, depths, meta = self.pre_process(image, depth, meta)

        meta['calib'] = calib
        meta['ground_plane'] = ground_plane

        images = images.to(self.opt.device)
        depths = depths.to(self.opt.device)

        output, dets, forward_time = self.process(
            images, depths, meta, return_time=True)

        # if self.opt.debug >= 2:
        #   self.debug(debugger, images, dets, output, scale)

        dets = self.post_process(dets, meta)

        if self.opt.debug >= 1:
            self.show_results(debugger, image, dets, calib_numpy)

        return {'results': dets, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
