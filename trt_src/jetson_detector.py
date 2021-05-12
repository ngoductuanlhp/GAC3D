from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

import time
import torch
import threading

from engine import Engine
# from engine_thread import EngineThread

from images import get_affine_transform
from decode import car_pose_decode


def gen_ground(P2, h, w, P0 = None):
    baseline = 0.54
    relative_elevation = 1.65
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
        self.batch = 1
        self.max_obj = 16

        self.video = args.video

        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
            [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0).expand(self.batch, self.max_obj, -1,-1).to(self.device)

        self.engine = Engine(args.load_model, args.dcn_lib)

        self.meta = None
        self.trans_input = None
        self.calib_np = None
        if self.video:
            self.img0_path = os.path.join(args.data_dir, 'sequences',args.demo,'image','0000000000.png')
            self.calib0_path = os.path.join(args.data_dir, 'sequences',args.demo, 'calib_cam_to_cam.txt')
            self.meta, self.trans_input = self.read_img_calib_video()

    def read_img_calib_video(self):
        img_0 = cv2.imread(self.img0_path)
        h, w = img_0.shape[:2]
        c = np.array([w/2., h/2.], dtype=np.float32)
        s = np.array([w, h], dtype=np.float32)
        
        trans_input = get_affine_transform(c, s, 0, [1280,384])
        
        trans_output_inv = get_affine_transform(c, s, 0, [320, 96],inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv).to(self.device)
        
        calib_file = open(self.calib0_path, 'r')
        for i, line in enumerate(calib_file):
            if line.split(':')[0] == 'P_rect_02':
                self.calib_np = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                self.calib_np = self.calib_np.reshape(3, 4)
                break

        ground_plane = gen_ground(self.calib_np, h, w).to(self.device)
        calib = torch.from_numpy(self.calib_np).to(self.device)
        
        meta = {'out_height': 384 // 4,
                'out_width': 1280 // 4,
                'trans_output_inv': trans_output_inv,
                'c': c,
                's': s,
                'ground_plane': ground_plane,
                'calib': calib
                }
        return meta, trans_input

    def run(self, img, calib):
        if self.video:
            img = self.preprocess_simple(img)
            meta = self.meta
        else:
            img, meta = self.preprocess(img, calib)
            meta['calib'] = meta['calib'].to(self.device)
            meta['trans_output_inv'] = meta['trans_output_inv'].to(self.device)
            meta['ground_plane'] = meta['ground_plane'].to(self.device)

        start_time = time.time()
        hm, hps, rot, dim, prob = self.engine(img)
        engine_time = time.time()
        dets = car_pose_decode(hm, hps, rot, dim, prob, K=self.max_obj, meta=meta, const=self.const)
        decode_time = time.time()
        # NOTE fetch from GPU to CPU
        dets = dets.squeeze(0).detach().cpu().numpy()

        engine_interval = engine_time - start_time
        decode_interval = decode_time - engine_time
        return dets, engine_interval, decode_interval

    def preprocess_simple(self, img):
        # img = cv2.warpAffine(img, self.trans_input, (1280,384), flags=cv2.INTER_LINEAR)
        # img = img.astype(np.float32)
        # img = np.transpose(img, [2, 0, 1])
        # img = np.ascontiguousarray(img)
        # img = np.ravel(img)
        return img

    def preprocess(self, img, calib):
        h, w = img.shape[:2]
        c = np.array([w/2., h/2.], dtype=np.float32)
        s = np.array([w, h], dtype=np.float32)
        
        trans_input = get_affine_transform(c, s, 0, [1280,384])
        
        img = cv2.warpAffine(img, trans_input, (1280,384), flags=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img = np.ascontiguousarray(img)
        img = np.ravel(img)
        
        trans_output_inv = get_affine_transform(c, s, 0, [320, 96],inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv)
        
        ground_plane = gen_ground(calib, h, w)
        calib = torch.from_numpy(calib)
        
        meta = {'out_height': 384 // 4,
                'out_width': 1280 // 4,
                'trans_output_inv': trans_output_inv,
                'c': c,
                's': s,
                'ground_plane': ground_plane,
                'calib': calib
                }
        return img, meta

    def read_image(self, img_name):
        # NOTE Read image
        img_path = os.path.join(self.img_dir , img_name + '.png')
        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        c = np.array([w/2., h/2.], dtype=np.float32)
        s = np.array([w, h], dtype=np.float32)
        
        trans_input = get_affine_transform(c, s, 0, [1280,384])
        img = cv2.warpAffine(img, trans_input, (1280,384), flags=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img = np.ascontiguousarray(img)
        img = np.ravel(img)
        return img, c, s, h, w

    def read_meta(self, img_name, c, s, h, w):
        trans_output_inv = get_affine_transform(c, s, 0, [320, 96],inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv).to(self.device)
        trans_output_inv = trans_output_inv.unsqueeze(0).unsqueeze(1).expand(self.batch, self.max_obj, -1, -1).contiguous().view(-1, 2, 3).float()

        # NOTE Read calib
        calib_path = os.path.join(self.calib_dir, img_name+'.txt')
        calib_file = open(calib_path, 'r')
        for i, line in enumerate(calib_file):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)

        # ground_plane = gen_ground(calib, h, w).to(self.device)
        P2_prime = calib
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
        ground_plane = z
        ground_plane = ground_plane.unsqueeze(0).unsqueeze(1).expand(self.batch, self.max_obj, -1).contiguous()

        calib = torch.from_numpy(calib).to(self.device).unsqueeze(0)
        meta = {'out_height': 96,
                'out_width': 320,
                'trans_output_inv': trans_output_inv,
                'c': c,
                's': s,
                'ground_plane': ground_plane,
                'calib': calib
                }
        return meta

    def read_img_and_meta(self, img_name):
        # NOTE Read image
        img_path = os.path.join(self.img_dir, img_name + '.png')
        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        c = np.array([w/2., h/2.], dtype=np.float32)
        s = np.array([w, h], dtype=np.float32)
        
        trans_input = get_affine_transform(c, s, 0, [1280,384])
        img = cv2.warpAffine(img, trans_input, (1280,384), flags=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img = np.ascontiguousarray(img)
        img = np.ravel(img)
        
        trans_output_inv = get_affine_transform(c, s, 0, [320, 96],inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv).to(self.device)
        trans_output_inv = trans_output_inv.unsqueeze(0).unsqueeze(1).expand(self.batch, self.max_obj, -1, -1).contiguous().view(-1, 2, 3).float()

        # NOTE Read calib
        calib_path = os.path.join(self.calib_dir, img_name+'.txt')
        calib_file = open(calib_path, 'r')
        for i, line in enumerate(calib_file):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)

        ground_plane = gen_ground(calib, h, w).to(self.device)
        ground_plane = ground_plane.unsqueeze(0).unsqueeze(1).expand(self.batch, self.max_obj, -1).contiguous()

        calib = torch.from_numpy(calib).to(self.device).unsqueeze(0)
        meta = {'out_height': 384 // 4,
                'out_width': 1280 // 4,
                'trans_output_inv': trans_output_inv,
                'c': c,
                's': s,
                'ground_plane': ground_plane,
                'calib': calib
                }
        return img, meta

