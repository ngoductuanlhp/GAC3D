import numpy as np
import cv2
import time
import os
import threading
import argparse
import torch
import shutil
from tqdm import tqdm

import pycuda.driver as cuda

from debugger import Debugger
from jetson_detector import JetsonDetector


DATA_DIR = "/home/ml4u/RTM3Dv2/kitti_format/data/kitti"
FILE_DEMO = "/home/ml4u/RTM3Dv2/kitti_format/data/kitti/test_jetson.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default='/home/ml4u/RTM3D_weights/res18_gac_base_fp16.trt',
                                 help='path to pretrained model')
    parser.add_argument('--data_dir', default='./kitti_format/data/kitti',
                                 help='path to dataset')
    parser.add_argument('--dcn_lib', default='/home/ml4u/GAC3D/trt_plugin/libDCN.so',
                                 help='path to DCN.so file')
    parser.add_argument('--demo', default='./kitti_format/data/kitti/test_jetson.txt',
                                 help='demo set')
    parser.add_argument('--result_dir', default='./kitti_format/exp/results_test',
                                 help='result dir')
    args = parser.parse_args()

    if os.path.exists(args.result_dir):
        shutil.rmtree(args.result_dir, True)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    debugger = Debugger(dataset='kitti_hp', ipynb=False, theme='white')
    detector = JetsonDetector(args=args)

    with open(args.demo, 'r') as f:
        lines = f.readlines()

    files = [img[:6] + '.png' for img in lines]

    start_count = 40
    start_idx = 0

    eng_time_arr, decode_time_arr, total_time_arr = [], [], []

    for f in tqdm(files):
        dets, time_dict = detector.run(f)

        # debugger.add_img(img_copy, img_id='car_pose')
        for bbox in dets:
            if bbox[4] > 0.3:
                # print("Save")
                # debugger.add_coco_bbox(bbox[:4], bbox[40], bbox[4], img_id='car_pose')
                # debugger.add_kitti_hp(bbox[5:23], img_id='car_pose')
                # debugger.add_bev(bbox, img_id='car_pose',is_faster=True)
                # calib_np = calib.cpu().numpy().squeeze(0)
                # debugger.add_3d_detection(bbox, calib_np, img_id='car_pose')
                debugger.save_kitti_format(bbox,f,result_dir=args.result_dir)
        # debugger.show_all_imgs(pause=True)
        total_interval = time_dict['total']
        eng_interval = time_dict['engine']
        decode_interval = time_dict['decode']
        print("Total: {:.3}s| Engine: {:.3}s| Decode: {:.3}s".format(total_interval, eng_interval, decode_interval))
        if start_idx > start_count:
            eng_time_arr.append(eng_interval)
            decode_time_arr.append(decode_interval)
            total_time_arr.append(total_interval)
        start_idx += 1

    eng_time_arr = np.array(eng_time_arr)
    decode_time_arr = np.array(decode_time_arr)
    total_time_arr = np.array(total_time_arr)
    
    # total:
    mean_val = np.mean(total_time_arr)
    median_val = np.median(total_time_arr)
    std = np.std(total_time_arr)
    print("Total time: Mean: {:.3}s| Median: {:.3}s| Std: {:.3}s| Max: {:.3}s| Min: {:.3}s".format(mean_val, median_val, std, np.max(total_time_arr), np.min(total_time_arr)))

    mean_val = np.mean(eng_time_arr)
    median_val = np.median(eng_time_arr)
    std = np.std(eng_time_arr)
    print("Engine time: Mean: {:.3}s| Median: {:.3}s| Std: {:.3}s| Max: {:.3}s| Min: {:.3}s".format(mean_val, median_val, std, np.max(eng_time_arr), np.min(eng_time_arr)))

    mean_val = np.mean(decode_time_arr)
    median_val = np.median(decode_time_arr)
    std = np.std(decode_time_arr)
    print("Decode time: Mean: {:.3}s| Median: {:.3}s| Std: {:.3}s| Max: {:.3}s| Min: {:.3}s".format(mean_val, median_val, std, np.max(decode_time_arr), np.min(decode_time_arr)))



main()
