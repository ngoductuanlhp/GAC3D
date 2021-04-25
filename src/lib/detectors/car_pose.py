from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode, _topk
from models.decode import car_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.post_process import car_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CarPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(CarPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx
        self.not_depth_guide = opt.not_depth_guide
        self.backbonea_arch = opt.arch.split('_')[0]

    def process(self, images, depths, meta, return_time=False):
        with torch.no_grad():
            if self.not_depth_guide or self.backbonea_arch == 'dla':
                output = self.model(images)[-1]
            else:
                output = self.model(images, depths)[-1]
            # output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            output['hps_var'] = output['hps_var'].sigmoid_()

            dets = car_pose_decode(
                output['hm'], output['hps'], output['hps_var'], output['dim'], output['rot'],
                reg=output['reg'], wh=output['wh'], K=self.opt.K, meta=meta, const=self.const,
                dynamic_dim=self.opt.dynamic_dim, axis_head_angle=self.opt.axis_head_angle, not_joint_task=self.opt.not_joint_task)

        if return_time:
            return output, dets, 0
        else:
            return output, dets

    def preprocess_depth(self, depth):
        n = 40
        delta = 2 * 80 / (n * (n + 1))
        depth = 1 + 8 * (depth) / delta
        depth = -0.5 + 0.5 * np.sqrt(depth)  # 0 -> 40
        depth = depth / 40  # 0 -> 1
        return depth

    def pre_process(self, image, depth, meta=None):
        height, width = image.shape[0:2]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = np.array([width, height], dtype=np.float32)

        trans_input = get_affine_transform(
            c, s, 0, [self.opt.input_w, self.opt.input_h])
        inp_image = cv2.warpAffine(
            image, trans_input, (self.opt.input_w, self.opt.input_h), flags=cv2.INTER_LINEAR)

        inp_image = (inp_image / 255).astype(np.float32)
        inp_image = (inp_image - self.mean) / self.std

        images = inp_image.transpose(2, 0, 1).reshape(
            1, 3, self.opt.input_h, self.opt.input_w)
        images = torch.from_numpy(images)

        # FIXME test depth
        # print(resized_depth.shape)
        # dummy_depth = np.random.randint(0, 10000, size = (new_height, new_width)).astype(np.uint16)
        # resized_depth = dummy_depth
        # print(resized_depth)
        # dummy_depth = np.ones_like(resized_depth) * 10 * 256
        # s = resized_depth.shape
        # resized_depth = np.random.randn(new_width, new_height, 1)
        # resized_depth = dummy_depth

        # resized_depth = np.arange(new_width * new_height).reshape(new_height,new_width)
        # resized_depth = np.clip(resized_depth, 0, 255 * 100)
        # print(resized_depth.shape)
        # resized_depth = cv2.resize(depth, (new_width, new_height))
        inp_depth = cv2.warpAffine(
            depth, trans_input, (self.opt.input_w, self.opt.input_h),
            flags=cv2.INTER_LINEAR)

        inp_depth = inp_depth.astype(np.float32) / 256.0
        # NOTE test new depth preproc
        # inp_depth = self.preprocess_depth(inp_depth)
        inp_depth = inp_depth[:, :, np.newaxis]
        inp_depth = (inp_depth - self.depth_mean) / self.depth_std
        # print(np.max(inp_depth), np.min(inp_depth))
        # inp_depth = inp_depth * 10000
        depths = inp_depth.transpose(2, 0, 1).reshape(
            1, 1, self.opt.input_h, self.opt.input_w)
        depths = torch.from_numpy(depths)

        meta = {'c': c, 's': s,
                'out_height': self.opt.input_h // self.opt.down_ratio,
                'out_width': self.opt.input_w // self.opt.down_ratio}

        trans_output_inv = get_affine_transform(
            c, s, 0, [meta['out_width'], meta['out_height']], inv=1)
        trans_output_inv = torch.from_numpy(
            trans_output_inv).unsqueeze(0).to(self.opt.device)
        meta['trans_output_inv'] = trans_output_inv
        return images, depths, meta

    def post_process(self, dets, meta):
        dets = dets.squeeze(0).detach().cpu().numpy()  # for batch size 1
        return dets

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
            img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

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
