import os
import cv2
import numpy as np
import time
import threading
import math
from utils import compute_box_3d, project_to_image

COLOR_LIST = [(0,255,0), (0,255,255), (255,0,255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
DETECTION_CATE = ['Car', 'Pedestrian', 'Cyclist']

def filter_det(cat, dim, pos, score_2d, score_3d):
    if cat == 0 and (dim[0] > 3 or dim[1] > 3 or dim[2] > 7 or dim[0] < 1.2 or dim[1] < 1.2 or dim[2] < 2):
        return True
    if dim[0] <= 0 or dim[1]<=0 or dim[2]<=0 or pos[2] >= 55 or pos[2] <= 0:
        return True
    if score_2d < 0.3 or score_3d < 0.3:
        return True
    return False

def save_kitti_format(results, img_path, result_dir):
    file_number = img_path.split('.')[-2][-6:]
    box = results[:4]
    score_2d = results[4]
    score_3d = results[12]
    score = score_2d * score_3d
    dim = results[5:8]
    pos = results[9:12]
    ori = results[8]
    cat = int(results[13])

    if filter_det(cat, dim, pos, score_2d, score_3d):
        return 
    
    write_detection_results(DETECTION_CATE[cat],result_dir,file_number,box,dim,pos,ori,score)
  
def write_detection_results(cls, result_dir, file_number, box,dim,pos,ori,score):
    '''One by one write detection results to KITTI format label files.
    '''
    if result_dir is None: return

    Px = pos[0]
    Py = pos[1]
    Pz = pos[2]
    l =dim[2]
    h = dim[0]
    w = dim[1]
    Py=Py+h/2
    pi=np.pi
    if ori > 2 * pi:
        while ori > 2 * pi:
            ori -= 2 * pi
    if ori < -2 * pi:
        while ori < -2 * pi:
            ori += 2 * pi

    if ori > pi:
        ori = 2 * pi - ori
    if ori < -pi:
        ori = 2 * pi + pi

    alpha = ori - math.atan2(Px, Pz)
    # convert the object from cam2 to the cam0 frame

    output_str = cls + ' '
    output_str += '%.2f %.d ' % (-1, -1)
    output_str += '%.7f %.7f %.7f %.7f %.7f ' % (alpha, box[0], box[1], box[2], box[3])
    output_str += '%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f \n' % (h, w, l, Px, Py, \
                                                                Pz, ori, score)
    # output_str += '%.2f %.2f %.2f %.2f %.2f ' % (alpha, box[0], box[1], box[2], box[3])
    # output_str += '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % (h, w, l, Px, Py, \
    #                                                               Pz, ori, score)

    # Write TXT files
    pred_filename = result_dir + '/' + file_number + '.txt'
    with open(pred_filename, 'a') as det_file:
        det_file.write(output_str)

class DisplayThread(threading.Thread):
    def __init__(self, queue, scale = 1):
        super(DisplayThread, self).__init__()
        self.running = True
        self.img = None
        self.queue = queue
        self.color_list = COLOR_LIST
        self.scale = scale

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.out_video = cv2.VideoWriter('dla34_test1.avi', fourcc, 10.0, (1280,340))

    def run(self):
        time_fps = time.time()
        while self.running:
            inputs = self.queue.get(block=True)
            fps = int(1 / (time.time() - time_fps))
            time_fps = time.time()
            dets = inputs['dets']
            calib = inputs['calib']
            self.img = inputs['img']

            if self.scale > 1:
                new_w, new_h = self.img.shape[1] // self.scale, self.img.shape[0] // self.scale
                self.img = cv2.resize(self.img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # self.im_bev = vis_utils.vis_create_bev(width=self.img.shape[0] * 2)

            for det in dets:
                dim = det[5:8]
                pos = det[9:12]
                ori = det[8]
                cat = int(det[13])
                score_2d = det[4]
                score_3d = det[12]
                
                if filter_det(cat, dim, pos, score_2d, score_3d):
                    continue
                    
                pos[1] = pos[1] + dim[0] / 2
                color = self.color_list[cat]
                box_3d = compute_box_3d(dim, pos, ori)
                box_2d = project_to_image(box_3d, calib)
                if self.scale > 1:
                    box_2d = box_2d // self.scale
                self.draw_projected_box3d(self.img, box_2d, color, front=True)
                
                
                # l = dim[2]
                # h = dim[0]
                # w = dim[1]
                # vis_utils.vis_box_in_bev(self.im_bev, pos, [l,h,w], ori,
                #                         score=det[12],
                #                         width=self.img.shape[0] * 2, gt='g')
            
            # cv2.putText(self.img, 'FPS: {}'.format(fps), (40, 40), FONT, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("2D", self.img)
            # img2 = cv2.resize(self.img, (1280,340))
            # self.out_video.write(img2)
            # cv2.imshow("BEV", self.im_bev)
            k = cv2.waitKey(1)
            if k == ord('q'):
                self.running = False

    def draw_projected_box3d(self, image, qs, color=(255, 255, 255), thickness=1, front=False):
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        # draw front view
        if front:
            cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[5, 0], qs[5, 1]), color, thickness, cv2.LINE_AA)
            cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[4, 0], qs[4, 1]), color, thickness, cv2.LINE_AA)

    def stop(self):
        self.out_video.release()
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