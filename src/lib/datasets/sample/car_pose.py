from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math




class CarPoseDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def read_calib(self, calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def preprocess_depth(self, depth):
        n = 40
        delta = 2 * 80 / (n * ( n + 1))
        depth = 1 + 8 * (depth)/delta
        depth = -0.5 + 0.5 * np.sqrt(depth) # 0 -> 40
        depth = depth / 40 # 0 -> 1
        return depth

    def get_orien(self, orien):
        heading = 0
        cls = 0
        offset = 0
        
        if abs(np.cos(orien)) > abs(np.sin(orien)):
            cls= 1
        else:
            cls= 0
        if cls == 0:
            if orien < 0:
              orien += np.pi
              heading= 1
            else:
              heading= 0
              orien = orien
            offset = orien - np.pi / 2
        else:
            if orien > np.pi / 2:
                heading= 1
                orien -= np.pi
            elif orien < -np.pi/2:
                heading = 1
                orien += np.pi
            else:
                heading = 0 
            offset = orien
        return cls, heading, offset

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        depth_path = os.path.join(self.depth_dir, file_name)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        label_sel = np.array([1.], dtype=np.float32)
        name_in = int(file_name[:6])
        if name_in > 14961 and name_in < 22480:
            label_sel[0] = 0.
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (width, height))
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = np.array([img.shape[1], img.shape[0]], dtype=np.int32)
        rot = 0
        flipped = False

        if self.split == 'train' and np.random.random() <= self.opt.affine:
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(
                    low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(
                    low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += img.shape[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += img.shape[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                depth = depth[:, ::-1]
                c[0] = width - c[0] - 1
                

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_w, self.opt.input_h])

        depth_inp = cv2.warpAffine(depth, trans_input,
                                   (self.opt.input_w, self.opt.input_h),
                                   flags=cv2.INTER_LINEAR)

        depth_inp = (depth_inp.astype(np.float32) / 256.0)
        depth_inp = depth_inp[:, :, np.newaxis]
        depth_inp = (depth_inp - self.depth_mean) / self.depth_std
        depth_inp = depth_inp.transpose(2, 0, 1)
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std

        inp = inp.transpose(2, 0, 1)

        num_joints = self.num_joints
        trans_output = get_affine_transform(
            c, s, 0, [self.opt.output_w, self.opt.output_h])
        trans_output_inv = get_affine_transform(
            c, s, 0, [self.opt.output_w, self.opt.output_h], inv=1)
        hm = np.zeros((self.num_classes, self.opt.output_h,
                       self.opt.output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dim = np.zeros((self.max_objs, 3), dtype=np.float32)
        location = np.zeros((self.max_objs, 3), dtype=np.float32)
        dep = np.zeros((self.max_objs, 1), dtype=np.float32)
        ori = np.zeros((self.max_objs, 1), dtype=np.float32)

        # rotbin = np.zeros((self.max_objs, 1), dtype=np.int64)
        # rotheading = np.zeros((self.max_objs, 1), dtype=np.int64)
        # rotres = np.zeros((self.max_objs, 1), dtype=np.float32)
        rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objs, 2), dtype=np.float32)

        rot_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros(
            (self.max_objs, self.num_joints * 2), dtype=np.uint8)

        rot_scalar = np.zeros((self.max_objs, 1), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian
        calib = np.array(anns[0]['calib'], dtype=np.float32)
        calib = np.reshape(calib, (3, 4))
        if flipped:
          calib[0, 2] = width - calib[0, 2] - 1

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1

            # one class training
            # if cls_id > 0:
            #     continue
            # pts = np.array(ann['keypoints'][:27], np.float32).reshape(num_joints, 3)
            pts = np.array(ann['keypoints'][:30],
                           np.float32).reshape(num_joints, 3)
            alpha1 = ann['alpha']
            orien = ann['rotation_y']
            loc = ann['location']
            if flipped:
                alpha1 = np.sign(alpha1) * np.pi - alpha1
                orien = np.sign(orien) * np.pi - orien
                loc[0] = -loc[0]
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

            while alpha1 >= np.pi:
                alpha1 -= 2 * np.pi
            while alpha1 < -np.pi:
                alpha1 += 2 * np.pi

            while orien >= np.pi:
                orien -= 2 * np.pi
            while orien < -np.pi:
                orien += 2 * np.pi

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            temp_center = [(bbox[0] + bbox[2])/2, (bbox[1]+ bbox[3])/ 2]

            skipped = False
            if temp_center[0] <= 0 or temp_center[0] >= self.opt.output_w:
              skipped = True
            if temp_center[1] <= 0 or temp_center[1] >= self.opt.output_h:
              skipped = True
            if loc[2] > 60:
              skipped = True

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if ((h > 0 and w > 0) or (rot != 0)) and not skipped:
                alpha = self._convert_alpha(alpha1)

                # axis-heading angle
                # alpha_cls, alpha_heading, alpha_offset = self.get_orien(alpha)
                # rotheading[k] = alpha_heading
                # rotbin[k] = alpha_cls
                # rotres[k] = alpha_offset

                # multi-bins
                if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                    rotbin[k, 0] = 1
                    rotres[k, 0] = alpha - (-0.5 * np.pi)
                if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                    rotbin[k, 1] = 1
                    rotres[k, 1] = alpha - (0.5 * np.pi)
                

                rot_scalar[k] = alpha
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(
                    0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
                reg[k] = ct - ct_int
                dim[k] = ann['dim']

                # regress dim offset
                # dim[k][0] = math.log(dim[k][0] / 1.63)
                # dim[k][1] = math.log(dim[k][1] / 1.53)
                # dim[k][2] = math.log(dim[k][2] / 3.88)
                dep[k] = loc[2]
                ori[k] = orien
                location[k] = loc
                reg_mask[k] = 1

                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0
                rot_mask[k] = 1
                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                # kps_cent[k,:]=pts[8,:2]
                for j in range(num_joints):
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output)
                    kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                    kps_mask[k, j * 2: j * 2 + 2] = 1

                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

        meta = {'file_name': file_name}
        
        ret = {'input': inp, 'depth': depth_inp,
               'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask, 'dim': dim, 'rotbin': rotbin, 'rotres': rotres, 
               'rot_mask': rot_mask, 'dep': dep, 'rotscalar': rot_scalar, 'calib': calib,
               'opinv': trans_output_inv, 'meta': meta, "label_sel": label_sel, 'location': location, 'ori': ori}
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        # if self.opt.hm_hp:
        #     ret.update({'hm_hp': hm_hp})
        # if self.opt.reg_hp_offset:
        #     ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        # if self.opt.debug > 0 or not self.split == 'train':
        #     gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #         np.zeros((1, 40), dtype=np.float32)
        #     meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        #     ret['meta'] = meta
        return ret
