import time
import os
import argparse
import shutil
from progress.bar import Bar
import threading
import cv2
# import multiprocessing
import queue

from jetson_detector import JetsonDetector
from utils_thread import ReadIOThread, DisplayThread
from utils_thread import save_kitti_format
from utils import AverageMeter

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_model', default='/home/ml4u/RTM3D_weights/res18_gac_base_200.trt',
    #                              help='path to pretrained model')
    parser.add_argument('--load_model', default='/home/ml4u/RTM3D_weights/dla34_last.trt',
                                 help='path to pretrained model')
    parser.add_argument('--data_dir', default='./kitti_format/data/kitti',
                                 help='path to dataset')
    parser.add_argument('--dcn_lib', default='/home/ml4u/GAC3D/trt_src/onnx-tensorrt/plugin/build/libDCN.so',
                                 help='path to DCN.so file')
    parser.add_argument('--demo', default='./kitti_format/data/kitti/val.txt',
                                 help='demo set')
    parser.add_argument('--result_dir', default='./kitti_format/exp/results_test',
                                 help='result dir')
    parser.add_argument('--video', action='store_true',
                                 help='infer on sequence')
    parser.add_argument('--vis', action='store_true',
                                 help='visualize outputs')
    args = parser.parse_args()

    if os.path.exists(args.result_dir):
        shutil.rmtree(args.result_dir, True)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.video:
        path = os.path.join(args.data_dir, 'sequences',args.demo,'image')
        files = sorted(os.listdir(path))
        total_iter = len(files)
    else:
        with open(args.demo, 'r') as f:
            lines = f.readlines()

        files = [img[:6] + '.png' for img in lines]
        total_iter = len(files)

    time_meter = {t: AverageMeter() for t in ['total', 'read', 'engine', 'decode']}

    bar = Bar('3D detection', max=total_iter)

    # NOTE detector
    detector = JetsonDetector(args)

    # NOTE io_thread
    io_queue = queue.Queue(maxsize=1)
    eventStop = threading.Event()
    eventStop.clear()
    readIOThread = ReadIOThread(args, files, io_queue, eventStop)
    readIOThread.start()
    
    # NOTE display_thread
    if args.vis:
        display_queue = queue.Queue(maxsize=1)
        displayThread = DisplayThread(display_queue)
        displayThread.start()


    idx = 0
    while not eventStop.is_set():
        start_time = time.time()
        
        # NOTE get input from io_thread
        inputs = io_queue.get(block=True)
        img, calib, read_interval, f = inputs['image'], inputs['calib'], inputs['read'], inputs['file']
        
        # NOTE run detection
        dets, engine_interval, decode_interval = detector.run(img, calib)
        end_time = time.time()

        time_dict = {
            'total': end_time - start_time,
            'engine': engine_interval,
            'decode': decode_interval,
            'read': read_interval
        }

        if args.video:
            calib = detector.calib_np

        if args.vis:
            display_queue.put({'dets': dets, 'calib': calib, 'img': img}, block=True)

        for bbox in dets:
            save_kitti_format(bbox,f,result_dir=args.result_dir)

        if idx < 40:
            Bar.suffix = 'Skip first 40 iterations.'
        else:
            Bar.suffix = '[{0}/{1}]|Total: {total:} |Eta: {eta:} '.format(idx, total_iter, total=bar.elapsed_td, eta=bar.eta_td)
            for t, val in time_dict.items():
                time_meter[t].update(val)
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(t, time_meter[t].avg)
        bar.next()
        idx += 1
    
    
    print("\nFinish. Average time:")
    for t, meter in time_meter.items():
        print('\t{}: {:.4f} ms'.format(t, meter.avg))
    bar.finish()

    if args.vis:
        displayThread.stop()

    # eng_time_arr = np.array(eng_time_arr)
    # decode_time_arr = np.array(decode_time_arr)
    # total_time_arr = np.array(total_time_arr)
    
    # # total:
    # mean_val = np.mean(total_time_arr)
    # median_val = np.median(total_time_arr)
    # std = np.std(total_time_arr)
    # print("Total time: Mean: {:.3}s| Median: {:.3}s| Std: {:.3}s| Max: {:.3}s| Min: {:.3}s".format(mean_val, median_val, std, np.max(total_time_arr), np.min(total_time_arr)))

    # mean_val = np.mean(eng_time_arr)
    # median_val = np.median(eng_time_arr)
    # std = np.std(eng_time_arr)
    # print("Engine time: Mean: {:.3}s| Median: {:.3}s| Std: {:.3}s| Max: {:.3}s| Min: {:.3}s".format(mean_val, median_val, std, np.max(eng_time_arr), np.min(eng_time_arr)))

    # mean_val = np.mean(decode_time_arr)
    # median_val = np.median(decode_time_arr)
    # std = np.std(decode_time_arr)
    # print("Decode time: Mean: {:.3}s| Median: {:.3}s| Std: {:.3}s| Max: {:.3}s| Min: {:.3}s".format(mean_val, median_val, std, np.max(decode_time_arr), np.min(decode_time_arr)))


if __name__ == "__main__":
    main()
