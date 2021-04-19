from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
# from .utils import _transpose_and_gather_feat
import torch.nn.functional as F
import iou3d_cuda
from lib.utils import kitti_utils_torch as kitti_utils
import time
import numpy as np
import os
from tqdm import tqdm



def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = []
    for line in lines:
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        # if data[0] == 'Car':
        obj = {}
        obj['dim'] = data[8:11]
        obj['loc'] = data[11:14]
        obj['roty'] = data[14]
        obj['meta'] = data[:15]
        obj['suggest'] = data[15]
        objects.append(obj)
    return objects


def load_predict(file_name):
    lines = [line.rstrip() for line in open(file_name)]
    objects = []
    for line in lines:
        data = line.split(' ')
        # print(data)
        data[1:] = [float(x) for x in data[1:]]
        if data[0] != '':
            obj = {}
            obj['dim'] = np.array([data[8:11]])
            obj['loc'] = np.array([data[11:14]])
            obj['roty'] = np.array([data[14:15]])
            obj['meta'] = data[:16]
            obj['contact'] = data[-1]
            objects.append(obj)
    return objects


def load_box(file_name):
    lines = [line.rstrip() for line in open(file_name)]
    box_info_array = [10,11,12,7,8,9,13,14]
    objects = []
    for line in lines:
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        box_info = np.array(data[1:])[box_info_array]
        objects.append(box_info)
    return np.array(objects)


def write_output(file_name, meta):
    output_str = meta[0] + ' '
    output_str += '%.2f %.d ' % (-1, -1)
    output_str += '%.7f %.7f %.7f %.7f %.7f ' % (
        meta[3], meta[4], meta[5], meta[6], meta[7])
    output_str += '%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f \n' % (
        meta[8], meta[9], meta[10], meta[11], meta[12], meta[13], meta[14], meta[15])
    # output_str += '%.2f %.2f %.2f %.2f %.2f ' % (alpha, box[0], box[1], box[2], box[3])
    # output_str += '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % (h, w, l, Px, Py, \
    #                                                               Pz, ori, score)
    with open(file_name, 'a') as det_file:
        det_file.write(output_str)


def non_max_suppression_fast(boxes, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    # x1 = boxes[:, 0]
    # y1 = boxes[:, 1]
    # x2 = boxes[:, 2]
    # y2 = boxes[:, 3]
    box_info = torch.from_numpy(boxes[:,:-1]).float()
    box_score = torch.from_numpy(boxes[:,-1]).float()
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(box_score)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        if len(idxs) == 1:
          break
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        # xx1 = np.maximum(x1[i], x1[idxs[:last]])
        # yy1 = np.maximum(y1[i], y1[idxs[:last]])
        # xx2 = np.minimum(x2[i], x2[idxs[:last]])
        # yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # # compute the width and height of the bounding box
        # w = np.maximum(0, xx2 - xx1 + 1)
        # h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        # overlap = (w * h) / area[idxs[:last]]
        # print('-------------')
        # print(idxs)
        overlap = boxes_iou3d_gpu(box_info[idxs[:last]].cuda(), box_info[i:i+1].cuda()).cpu().numpy()
        # print(overlap)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")
    return pick


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """
    ans_iou = torch.cuda.FloatTensor(torch.Size(
        (boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(
        boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size(
        (boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(
        boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    # print(boxes_a[:, 1], boxes_a[:, 3])
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x)
                                      for x in value.split()]).astype(np.float32)
            except ValueError:
                pass

    return data['P2'].reshape((3, 4))


if __name__ == "__main__":
    # label_dir = '/home/quan/Thesis-Ref-Repo/RTM3Dv2/kitti_format/data/kitti/label'
    # label_dir = '/home/quan/Thesis-Ref-Repo/RTM3Dv2/kitti_format/exp/new_label'
    # predict_dir = '/home/quan/Thesis-Ref-Repo/RTM3Dv2/kitti_format/exp/results/data'
    # results_dir = '/home/tuan/RTM3Dv2/kitti_format/exp/results_refine_new/data'
    # new_results_dir = '/home/tuan/RTM3Dv2/kitti_format/exp/results_nms_new'
    results_dir = '/home/tuan/RTM3Dv2/kitti_format/exp/results_refine_2d3d/data'
    new_results_dir = '/home/tuan/RTM3Dv2/kitti_format/exp/results_nms_2d3d'

    test = ['000008.txt']
    # test = ['000002.txt']

    # dir = 'path/to/dir'
    for f in os.listdir(new_results_dir):
        os.remove(os.path.join(new_results_dir, f))

    total_old = 0
    total_new = 0

    for image in tqdm(os.listdir(results_dir)):
    # for image in test:
        predict = os.path.join(results_dir, image)
        pred_objects = load_predict(predict)
        boxes = load_box(predict)
        
        keep = non_max_suppression_fast(boxes)
        keep = [k.item() for k in keep]
        results_file = os.path.join(new_results_dir, image)
        # # print(keep)
        if len(keep) < len(pred_objects):
          print('Keep', keep)
          print("origin", len(pred_objects))
          print(image)

        total_old += len(pred_objects)
        # total_new += len(keep)
        if len(pred_objects) == 0:
            with open(results_file, 'a') as det_file:
                det_file.write('')
        for i in range(len(pred_objects)):
            # if i in keep:
            #     print("True")
            if i in keep and pred_objects[i]['meta'][-1] > 0.6:
                meta = pred_objects[i]['meta']
                write_output(results_file, meta)
                total_new += 1
        # obj1 = pred_objects[0]
        # obj2 = pred_objects[1]

        # gt_box = torch.cat((torch.from_numpy(obj1['loc']), torch.from_numpy(
        #     obj1['dim']), torch.from_numpy(obj1['roty'])), dim=1).float()
        # pred_box = torch.cat((torch.from_numpy(obj2['loc']), torch.from_numpy(
        #     obj2['dim']), torch.from_numpy(obj2['roty'])), dim=1).float()
        # print(boxes_iou3d_gpu(pred_box.cuda(), gt_box.cuda()))

    print(total_old, total_new)