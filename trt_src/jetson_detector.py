from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from rtm3d_engine import RTM3D_Engine
from debugger import Debugger

from images import get_affine_transform
from decode import car_pose_decode


def gen_ground(P2_prime, h, w, P0 = None):
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

class JetsonDetector(object):
    def __init__(self, args):
        self.device = torch.device('cuda')
        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
            [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0).to(self.device)


        self.img_dir = os.path.join(args.data_dir, "image")
        self.calib_dir = os.path.join(args.data_dir, "calib")
        self.engine = RTM3D_Engine(args.load_model, args.dcn_lib)
        self.max_obj = 10

    
    def run(self, img_file):
        # NOTE read img and meta-data
        img, meta = self.read_img_and_meta(img_file)

        start_time = time.time()
        # NOTE run model
        hm, hps, rot, dim, prob = self.engine(img)
        engine_time = time.time()

        # NOTE 2D-3D transformation
        dets = car_pose_decode(hm, hps, dim, rot, prob, K=self.max_obj, meta=meta, const=self.const)
        decode_time = time.time()

        time_dict = {
            'total': decode_time - start_time,
            'engine': engine_time - start_time,
            'decode': decode_time - engine_time
        }
        # NOTE post process
        dets = dets.squeeze(0).detach().cpu().numpy()

        return dets, time_dict

    def read_img_and_meta(self, img_file):
        
        # NOTE Read image
        t = time.time()
        img_name = img_file.split('.')[0]
        img_path = os.path.join(self.img_dir , img_file)
        img = cv2.imread(img_path)
    #     read_time = time.time()
    #     print("Read time: {:.3}".format(read_time - t))

        h, w = img.shape[:2]
        c = np.array([w/2., h/2.], dtype=np.float32)
        s = np.array([w, h], dtype=np.float32)
        
        # img = cv2.resize(img, (1280, 384)).astype(np.float32)
        trans_input = get_affine_transform(c, s, 0, [1280,384])
        img = cv2.warpAffine(img, trans_input, (1280,384), flags=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img = np.ascontiguousarray(img)
        img = np.ravel(img)
        
        trans_output_inv = get_affine_transform(c, s, 0, [320, 96],inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv).unsqueeze(0).unsqueeze(1).to(self.device)

        # NOTE Read calib
        calib_path = os.path.join(self.calib_dir, img_name+'.txt')
        calib_file = open(calib_path, 'r')
        for i, line in enumerate(calib_file):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)

        ground_plane = gen_ground(calib, h, w).unsqueeze(0).unsqueeze(1).to(self.device)
        calib = torch.from_numpy(calib).unsqueeze(0).to(self.device)
        meta = {'out_height': 384 // 4,
                'out_width': 1280 // 4,
                'trans_output_inv': trans_output_inv,
                'c': c,
                's': s,
                'ground_plane': ground_plane,
                'calib': calib
                }
        return img, meta


    def post_process(self, dets, meta):
        dets = dets.squeeze(0).detach().cpu().numpy()  # for batch size 1
        return dets


    def show_results(self, debugger, image, results, calib):
        debugger.add_img(image, img_id='car_pose')
        for bbox in results:
            if bbox[4] > self.opt.vis_thresh:
                # debugger.add_coco_bbox(bbox[:4], bbox[40], bbox[4], img_id='car_pose')
                # debugger.add_kitti_hp(bbox[5:23], img_id='car_pose')
                # debugger.add_bev(bbox, img_id='car_pose',is_faster=self.opt.faster)
                # debugger.add_3d_detection(bbox, calib, img_id='car_pose')
                debugger.save_kitti_format(
                    bbox, self.image_path, self.opt, img_id='car_pose')
        if self.opt.vis:
            debugger.show_all_imgs(pause=self.pause)
