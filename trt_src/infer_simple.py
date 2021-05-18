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
    parser.add_argument('--load_model', default='/home/ml4u/RTM3D_weights/dla34_e2e_int8.trt',
                                 help='path to pretrained model')
    #parser.add_argument('--load_model', default='/home/ml4u/RTM3D_weights/dla34_last.trt',
    #                             help='path to pretrained model')
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
    parser.add_argument('--save', action='store_true',
                                 help='save results to disk')                             
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
    readIOThread = ReadIOThread(args, files, io_queue, eventStop, detector.trans_input)
    # readIOThread.start()
    
    # NOTE display_thread
    display_queue = queue.Queue(maxsize=1)
    displayThread = DisplayThread(display_queue, args)
    # displayThread.start()

    idx = 0
    for f in files:
        img_name = f.split('.')[0]
        start_time = time.time()
        img, processed_img, calib = readIOThread.read_io(img_name)
        read_time = time.time()
        dets, engine_interval, decode_interval = detector.run(processed_img, calib)
        
        processed_dets = displayThread.postprocess(dets)
        end_time = time.time()

        time_dict = {
            'total': end_time - start_time,
            'engine': engine_interval,
            'decode': decode_interval,
            'read': read_time - start_time
        }

        if args.video:
            calib = detector.calib_np

        # outputs = {'dets': dets, 'calib': calib}
        # if args.vis:
        #     outputs['img'] = inputs['img']
        # if args.save:
        #     outputs['file'] = inputs['file']
        # display_queue.put(outputs, block=True)

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


if __name__ == "__main__":
    main()
