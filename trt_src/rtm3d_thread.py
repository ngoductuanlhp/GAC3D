import threading
import numpy as np

import pycuda.driver as cuda

import time
import os
import cv2

from rtm3d_engine import RTM3D_Engine

IMG_DIR_2 = "/home/ml4u/RTM3Dv2/demo_kitti_format/data/kitti/image/"
CALIB_DIR = "/home/ml4u/RTM3Dv2/demo_kitti_format/data/kitti/calib/"

class RTM3D_Thread(threading.Thread):
    """RTM3D_Threadd
    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the TrtThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, eventStart, eventEnd):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd
        
        self.eventStop = threading.Event()

        self.cuda_ctx = None  # to be created when run
        self.trt_model = None   # to be created when run

        self.img_file = None

        self.hm, self.hps, self.rot, self.dim, self.prob = None, None, None, None, None
        self.calib = None

    def read_img_and_calib(self, img_file):
        # NOTE Read image
        img_name = img_file.split('.')[0]
        img_path = os.path.join(IMG_DIR_2, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1280, 384))
        img = np.transpose(img, [2, 0, 1]).astype(np.float32)
        img = np.ascontiguousarray(img)

        # NOTE Read calib
        calib_path = os.path.join(CALIB_DIR,img_name+'.txt')
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
        # calib=torch.from_numpy(calib_numpy).unsqueeze(0).to(self.opt.device)
        return img, calib

    def run(self):
        """Run until 'running' flag is set to False by main thread.
        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = RTM3D_Engine()

        while not self.eventStop.is_set():
            self.eventStart.wait()
            self.eventStart.clear()
            if self.img_file is not None:
                print("File: ", self.img_file)
                img, calib = self.read_img_and_calib(self.img_file)
                self.calib = calib
                self.hm, self.hps, self.rot, self.dim, self.prob = self.trt_model(img)
                self.img_file = None
            self.eventEnd.set()

        del self.trt_model
        self.cuda_ctx.pop()
        del self.cuda_ctx

    def stop(self):
        self.eventStop.set()
        # self.join()