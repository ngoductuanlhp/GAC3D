import os
import cv2
import numpy as np
import time
import threading
import math
import math as m
from utils import compute_box_3d, project_to_image

COLOR_LIST = [(0,255,0), (0,255,255), (255,0,255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
DETECTION_CATE = ['Car', 'Pedestrian', 'Cyclist']

def filter_det(cat, dim, pos, score_2d, score_3d):
    # if cat == 0 and (dim[0] > 3 or dim[1] > 3 or dim[2] > 7 or dim[0] < 1.2 or dim[1] < 1.2 or dim[2] < 2):
    #     return True
    if dim[0] <= 0 or dim[1]<=0 or dim[2]<=0 or pos[2] >= 55 or pos[2] <= 0:
        return True
    if score_2d < 0.3:
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


def E2R(Ry, Rx, Rz):
    '''Combine Euler angles to the rotation matrix (right-hand)

        Inputs:
            Ry, Rx, Rz : rotation angles along  y, x, z axis
                        only has Ry in the KITTI dataset
        Returns:
            3 x 3 rotation matrix

    '''
    R_yaw = np.array([[m.cos(Ry), 0, m.sin(Ry)],
                    [0, 1, 0],
                    [-m.sin(Ry), 0, m.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, m.cos(Rx), -m.sin(Rx)],
                        [0, m.sin(Rx), m.cos(Rx)]])
    # R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0],
    #                    [m.sin(Rz), m.cos(Rz), 0],
    #                    [ 0,         0 ,     1]])
    return (R_pitch.dot(R_yaw))

def space2Bev(P0, side_range=(-20, 20),
            fwd_range=(0, 70),
            res=0.1):
    x_img = (P0[0] / res).astype(np.int32)
    y_img = (-P0[2] / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.floor(fwd_range[1] / res)) - 1

    return np.array([x_img, y_img])


def vis_box_in_bev(im_bev, pos, dims, orien, width=750, gt=False,score=None,
                side_range=(-20, 20), fwd_range=(0, 70),
                min_height=-2.73, max_height=1.27, color=(0,0,255)):
    ''' Project 3D bounding box to bev image for simply visualization
        It should use consistent width and side/fwd range input with
        the function: vis_lidar_in_bev

        Inputs:
            im_bev:         cv image
            pos, dim, orien: params of the 3D bounding box
        Return:
            cv color image

    '''
    dim = dims.copy()
    buf = dim.copy()
    # dim[0]=buf[2]
    # dim[2]=buf[0]
    # dim[0]=buf[1]
    # dim[1]=buf[2]
    # dim[2]=buf[0]
    res = float(fwd_range[1] - fwd_range[0]) / width

    R = E2R(orien, 0, 0)
    pts3_c_o = []
    pts2_c_o = []
    # pts3_c_o.append(pos + R.dot([-dim[0], 0, -dim[2]])/2.0)
    # pts3_c_o.append(pos + R.dot([-dim[0], 0, dim[2]])/2.0) #-x z
    # pts3_c_o.append(pos + R.dot([dim[0], 0, dim[2]])/2.0) # x, z
    # pts3_c_o.append(pos + R.dot([dim[0], 0, -dim[2]])/2.0)
    #
    # pts3_c_o.append(pos + R.dot([0, 0, dim[2]*2/3]))

    pts3_c_o.append(pos + R.dot(np.array([dim[0] / 2., 0, dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([dim[0] / 2, 0, -dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-dim[0] / 2, 0, -dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-dim[0] / 2, 0, dim[2] / 2.0]).T))

    pts3_c_o.append(pos + R.dot([dim[0] / 1.5, 0, 0]))
    pts2_bev = []
    for index in range(5):
        pts2_bev.append(space2Bev(pts3_c_o[index], side_range=side_range,
                                fwd_range=fwd_range, res=res))

    # if gt is False:
    #     lineColor3d = (100, 100, 0)
    # else:
    #     lineColor3d = (0, 0, 255)
    # if gt == 'next':
    #     lineColor3d = (255, 0, 0)
    # if gt == 'g':
    #     lineColor3d = (0, 255, 0)
    # if gt == 'b':
    #     lineColor3d = (255, 0, 0)
    # if gt == 'n':
    #     lineColor3d = (5, 100, 100)
    lineColor3d = color
    thick = 2
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, thick)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, thick)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, thick)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, thick)

    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, thick)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, thick)

    # if score is not None:
    #     show_text(im_bev,pts2_bev[4],score)
    # return im_bev

def vis_create_bev(width=750, side_range=(-20, 20), fwd_range=(0, 70),
                   min_height=-2.5, max_height=1.5):
    ''' Project pointcloud to bev image for simply visualization

        Inputs:
            pointcloud:     3 x N in camera 2 frame
        Return:
            cv color image

    '''
    res = float(fwd_range[1] - fwd_range[0]) / width
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    # im[:, :] = 255
    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # print(y_max, x_max)
    # print(im_rgb.shape)
    im_rgb = cv2.arrowedLine(im_rgb, (int(x_max/2), int(y_max-5)), (int(x_max/2), int(y_max-40)), (0, 0, 255), 4)
    return im_rgb


class DisplayThread(threading.Thread):
    def __init__(self, queue, args, scale = 1):
        super(DisplayThread, self).__init__()
        self.running = True
        self.img = None
        self.img_origin = None
        self.queue = queue
        self.color_list = COLOR_LIST
        self.scale = scale
        self.args = args

        # self.fig_3d = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.out_video = cv2.VideoWriter('dla34_test1.avi', fourcc, 10.0, (1280,384))

    def postprocess(self, dets):
        # dets n x 13
        # dim = det[5:8]
        # pos = det[9:12]
        # ori = det[8]
        # cat = int(det[13])
        # score_2d = det[4]
        # score_3d = det[12]
        conditions = np.concatenate((dets[:, 5:8] > 0.1, dets[:, [5]] < 3, dets[:, [6]] < 5, dets[:, [7]] < 8,
                                    dets[:, [4]] > 0.3, 
                                    dets[:, [11]] >= 0, dets[:, [11]] <= 55), axis=1)
        valid = np.bitwise_and.reduce(conditions, axis=1)
        processed_dets = dets[valid]
        return processed_dets


    def vis_3d(self, pc_velo):
        pass


    def run(self):
        # time_fps = time.time()
        time_out = 20
        while self.running:
            try:
                inputs = self.queue.get(block=True, timeout=time_out)
            except Exception as e:
                pass
            time_out = 1
            # fps = int(1 / (time.time() - time_fps))
            start_time = time.time()
            dets = inputs['dets']
            calib = inputs['calib']
            
            # NOTE post process
            # processed_dets = dets
            processed_dets = self.postprocess(dets)
            end_time = time.time()
            # print('\nPost time: {:.4f}'.format(end_time - start_time))

            # NOTE save results
            if self.args.save:
                for det in processed_dets:
                    save_kitti_format(det, inputs['file'], result_dir=self.args.result_dir)

            if self.args.vis:
                # pc_velo = inputs["pc_velo"]
                # draw_lidar(pc_velo, fig=self.fig_3d)

                self.img = inputs['img']
                self.img_origin = self.img.copy()

                if self.scale > 1:
                    new_w, new_h = self.img.shape[1] // self.scale, self.img.shape[0] // self.scale
                    self.img = cv2.resize(self.img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                self.im_bev = vis_create_bev(width=self.img.shape[0] * 2)

                for det in processed_dets:
                    dim = det[5:8]
                    pos = det[9:12]
                    ori = det[8]
                    cat = int(det[13])
                    score_2d = det[4]
                    score_3d = det[12]
                    
                    # if filter_det(cat, dim, pos, score_2d, score_3d):
                    #     continue
                        
                    pos[1] = pos[1] + dim[0] / 2
                    color = self.color_list[cat]
                    box_3d = compute_box_3d(dim, pos, ori)
                    box_2d = project_to_image(box_3d, calib)
                    if self.scale > 1:
                        box_2d = box_2d // self.scale
                    self.draw_projected_box3d(self.img, box_2d, color, front=True)
                    
                    
                    l = dim[2]
                    h = dim[0]
                    w = dim[1]
                    vis_box_in_bev(self.im_bev, pos, [l,h,w], ori,
                                            score=det[12],
                                            width=self.img.shape[0] * 2, gt='g', color=color)
                
                # cv2.putText(self.img, 'FPS: {}'.format(fps), (40, 40), FONT, 0.8, (0,0,255), 2, cv2.LINE_AA)
                self.img = cv2.resize(self.img, (1280,384))
                self.img_origin = cv2.resize(self.img_origin, (1280, 384))

                vis_img = np.concatenate((self.img_origin, np.ones((30,1280,3), np.uint8) * 255,self.img), axis=0)
                cv2.imshow("Image", vis_img)
                # cv2.imshow("Input", self.img_origin)
                cv2.imshow("Bird-eye-view", self.im_bev)
                # cv2.imwrite('test.png', self.img)
                # quit()
                
                # self.out_video.write(self.img)

                # mlab.show(1)
            
                k = cv2.waitKey(40)
                if k == ord('q'):
                    self.running = False
                # mlab.clf()
        # self.out_video.release()

    def draw_projected_box3d(self, image, qs, color=(255, 255, 255), thickness=1, front=False):
        qs = qs.astype(np.int32)

        max_box = np.max(qs, axis = 0)
        min_box = np.min(qs, axis = 0)
        if (max_box[0] - min_box[0] > 500) or (min_box[0] - min_box[0] > 200):
            return

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
        # self.out_video.release()
        self.running = False

# class ReadIOThread(multiprocessing.Process):
class ReadIOThread(threading.Thread):
    def __init__(self, args, demo_files, queue, eventStop, trans_input=None):
        super(ReadIOThread, self).__init__()

        if args.video:
            self.img_dir = os.path.join(args.data_dir, 'sequences',args.demo,'image')
        else:
            self.img_dir = os.path.join(args.data_dir, "image")

        self.calib_dir = os.path.join(args.data_dir, "calib")
        self.velo_dir = os.path.join(args.data_dir, 'sequences',args.demo, "velodyne")
        # self.device = device
        self.queue = queue
        self.eventStop = eventStop
        self.batch = 1
        self.max_obj = 10
        self.video = args.video

        self.demo_files = demo_files
        self.trans_input = trans_input

        self.args = args
        self.img_height, self.img_width = args.img_dim

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


    def run(self):
        for idx, f in enumerate(self.demo_files):
            img_name = f.split('.')[0]
            start_time = time.time()
            
            img, processed_img, calib = self.read_io(img_name)
            read_time = time.time()
            
            outputs = {'p_img': processed_img, 'calib': calib, 'read': read_time - start_time, 'file': f}
            if self.args.vis or not self.args.video:
                outputs['img'] = img
            self.queue.put(outputs, block=True)

        self.eventStop.set()

    def load_velo_scan(self, velo_filename):
        scan = np.fromfile(velo_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        # for i in range(0, scan.shape[0], 20):
        #     # if scan[i,2] >= 10:
        #     print(scan[i,:])
        return scan[:,:3]

    def read_io(self, img_name):
        # NOTE Read image
        img_path = os.path.join(self.img_dir, img_name + '.png')
        img = cv2.imread(img_path)

        processed_img = None
        if self.video:
            if self.args.use_torch:
                processed_img = self.preprocess_img_torch(img)
            else:
                processed_img = self.preprocess_img(img)
        
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

        velo_path = os.path.join(self.velo_dir, img_name+'.bin')                
        # pc_velo = self.load_velo_scan(velo_path)
        return img, processed_img, calib


    def preprocess_img(self, img, trans_input=None):
        if trans_input is None:
            trans_input = self.trans_input
        img = cv2.warpAffine(img, trans_input, (self.img_width,self.img_height), flags=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img = np.ascontiguousarray(img)
        img = np.ravel(img)
        return img

    def preprocess_img_torch(self, img, trans_input=None):
        if trans_input is None:
            trans_input = self.trans_input
        img = cv2.warpAffine(img, trans_input, (self.img_width,self.img_height), flags=cv2.INTER_LINEAR)
        
        img = (img / 255).astype(np.float32)
        img = (img - self.mean) / self.std

        img = img.transpose(2, 0, 1).reshape(
            1, 3, self.img_height, self.img_width)
        return img

    # def preprocess_img_calib(self, img, calib):
    #     h, w = img.shape[:2]
    #     c = np.array([w/2., h/2.], dtype=np.float32)
    #     s = np.array([w, h], dtype=np.float32)
        
    #     trans_input = get_affine_transform(c, s, 0, [1280,384])
        
    #     img = cv2.warpAffine(img, trans_input, (1280,384), flags=cv2.INTER_LINEAR)

    #     img = img.astype(np.float32)
    #     img = np.transpose(img, [2, 0, 1])
    #     img = np.ascontiguousarray(img)
    #     img = np.ravel(img)
        
    #     trans_output_inv = get_affine_transform(c, s, 0, [320, 96],inv=1)
    #     trans_output_inv = torch.from_numpy(trans_output_inv)
        
    #     ground_plane = gen_ground(calib, h, w)
    #     calib = torch.from_numpy(calib)
        
    #     meta = {'out_height': 384 // 4,
    #             'out_width': 1280 // 4,
    #             'trans_output_inv': trans_output_inv,
    #             'c': c,
    #             's': s,
    #             'ground_plane': ground_plane,
    #             'calib': calib
    #             }
    #     return img, meta