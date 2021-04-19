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
from models.decode import car_pose_decode,car_pose_decode_faster
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.post_process import car_pose_post_process, car_pose_post_process_single
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CarPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(CarPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images, depths, meta, return_time=False):
        with torch.no_grad():
            # torch.cuda.synchronize()
            if self.opt.base_model:
                output = self.model(images)[-1]
            else:
                output = self.model(images, depths)[-1]

            torch.cuda.synchronize()
            forward_time = time.time()

            dets = car_pose_decode_faster(
                output['hm'], output['hps'], output['dim'], output['rot'], output['prob'], reg=None, wh=None,K=self.opt.K, meta=meta, const=self.const)


        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta):
        # dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # dets: batch x K x dim
        dets = dets.detach().cpu().numpy().squeeze(0)
        dets = car_pose_post_process_single(
            dets, meta['trans_output_inv'])

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
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], bbox[40], bbox[4], img_id='car_pose')
                debugger.add_kitti_hp(bbox[5:23], img_id='car_pose')
                debugger.add_bev(bbox, img_id='car_pose',is_faster=self.opt.faster)
                debugger.add_3d_detection(bbox, calib, img_id='car_pose')
                debugger.save_kitti_format(bbox,self.image_path,self.opt,img_id='car_pose',is_faster=self.opt.faster)
        if self.opt.vis:
            debugger.show_all_imgs(pause=self.pause)

