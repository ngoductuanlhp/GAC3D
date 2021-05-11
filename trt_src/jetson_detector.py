from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

import time
import torch
import threading

from rtm3d_engine import RTM3D_Engine
from rtm3d_thread import RTM3D_Thread

from images import get_affine_transform
from decode import car_pose_decode

from ddd_utils import compute_box_3d
import vis_3d_utils as vis_utils



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

class DisplayThread(threading.Thread):
    def __init__(self, queue):
        super(DisplayThread, self).__init__()
        self.running = True
        self.img = None
        self.queue = queue
        self.color_list = [(0,255,0), (0,255,255), (255,0,255)]

    def run(self):
        while self.running:
            inputs = self.queue.get(block=True)
            dets = inputs['dets']
            calib = inputs['calib']
            self.img = inputs['img']

            self.im_bev = vis_utils.vis_create_bev(width=self.img.shape[0] * 2)

            for det in dets:
                dim = det[5:8]
                pos = det[9:12]
                ori = det[8]
                cat = int(det[13])
                if cat == 0 and (dim[0] > 3 or dim[1] > 3 or dim[2] > 7 or dim[0] < 1.2 or dim[1] < 1.2 or dim[2] < 2):
                    continue
                if dim[0] <= 0 or dim[1]<=0 or dim[2]<=0 or pos[2] >= 55 or pos[2] <= 0:
                    continue
                if det[4] < 0.3 or det[12] < 0.3:
                    continue
                    
                pos[1] = pos[1] + dim[0] / 2
                color = self.color_list[cat]
                box_3d = compute_box_3d(dim, pos, ori)
                box_2d = self.project_to_image(box_3d, calib)
                self.draw_box_3d(box_2d, color)
                # l = dim[2]
                # h = dim[0]
                # w = dim[1]
                # vis_utils.vis_box_in_bev(self.im_bev, pos, [l,h,w], ori,
                #                         score=det[12],
                #                         width=self.img.shape[0] * 2, gt='g')

            cv2.imshow("2D", self.img)
            # cv2.imshow("BEV", self.im_bev)
            k = cv2.waitKey(1)
            if k == ord('q'):
                self.running = False

    def draw_box_3d(self, corners, c):
        face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
        for ind_f in range(3, -1, -1):
            f = face_idx[ind_f]
            for j in range(4):
                cv2.line(self.img, (corners[f[j], 0], corners[f[j], 1]),
                        (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
            if ind_f == 0:
                cv2.line(self.img, (corners[f[0], 0], corners[f[0], 1]),
                        (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
                cv2.line(self.img, (corners[f[1], 0], corners[f[1], 1]),
                        (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
    
    def project_to_image(self, pts_3d, P):
        # pts_3d: n x 3
        # P: 3 x 4
        # return: n x 2
        pts_3d_homo = np.concatenate(
            [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
        pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        # import pdb; pdb.set_trace()
        return pts_2d

    def stop(self):
        self.running = False

# class ReadIOThread(multiprocessing.Process):
class ReadIOThread(threading.Thread):
    def __init__(self, args, demo_files, queue, eventStop):
        super(ReadIOThread, self).__init__()

        if args.video:
            self.img_dir = os.path.join(args.data_dir, 'sequences',args.demo,'image')
        else:
            self.img_dir = os.path.join(args.data_dir, "image")
        self.calib_dir = os.path.join(args.data_dir, "calib")
        # self.device = device
        self.queue = queue
        self.eventStop = eventStop
        self.batch = 1
        self.max_obj = 10
        self.video = args.video

        self.demo_files = demo_files


    def run(self):
        for idx, f in enumerate(self.demo_files):
            img_name = f.split('.')[0]
            start_time = time.time()
            
            img, calib = self.read_io(img_name)
            read_time = time.time()
            
            self.queue.put({'image': img, 'calib': calib, 'read': read_time - start_time, 'file': f}, block=True)

        self.eventStop.set()
        # self.join()

    def read_io(self, img_name):
        # NOTE Read image
        img_path = os.path.join(self.img_dir, img_name + '.png')
        img = cv2.imread(img_path)
        
        # NOTE Read calib
        if self.video:
            calib = None
        else:
            calib_path = os.path.join(self.calib_dir, img_name+'.txt')
            calib_file = open(calib_path, 'r')
            for i, line in enumerate(calib_file):
                if i == 2:
                    calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                    calib = calib.reshape(3, 4)
                    break

        
        return img, calib

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

        self.engine = RTM3D_Engine(args.load_model, args.dcn_lib)

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
        # NOTE post process
        dets = dets.squeeze(0).detach().cpu().numpy()

        engine_interval = engine_time - start_time
        decode_interval = decode_time - engine_time
        return dets, engine_interval, decode_interval

    def preprocess_simple(self, img):
        img = cv2.warpAffine(img, self.trans_input, (1280,384), flags=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img = np.ascontiguousarray(img)
        img = np.ravel(img)
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

