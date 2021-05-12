from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.start_count = 40

    def update(self, val, n=1):
        self.val = val
        self.count += n
        # if self.count < self.start_count:
        #     return

        self.sum += val * n
        if self.count > 0:
          self.avg = self.sum / (self.count)

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
      tmp[:, e[0], ...], tmp[:, e[1], ...] = \
        tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
      tmp[:, e[0], ...], tmp[:, e[1], ...] = \
        tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2,0]
    y_corners = [0,0,0,0,-h,-h,-h,-h,-h/2]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2,0]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d