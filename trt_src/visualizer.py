import open3d
from open3d import Visualizer
from open3d import PointCloud
from open3d import Vector3dVector
from open3d import read_pinhole_camera_parameters

from box3d import Box3D

import numpy as np
import os
import cv2

img_height = 375
img_width = 1242

small_img_height = 187
small_img_width = 620

DATA_DIR = "/home/tuan/GAC3D/kitti_format/data/kitti/sequences/2011_09_26_0009"

class ImgCreatorLiDAR:
    def __init__(self):
        self.counter = 0
        self.trajectory = read_pinhole_camera_parameters("/home/fregu856/3DOD_thesis/visualization/camera_trajectory.json") # NOTE! you'll have to adapt this for your file structure

    def move_forward(self, vis):
        # this function is called within the Visualizer::run() loop.
        # the run loop calls the function, then re-renders the image.

        if self.counter < 2: # (the counter is for making sure the camera view has been changed before the img is captured)
            # set the camera view:
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.trajectory)

            self.counter += 1
        else:
            # capture an image:
            img = vis.capture_screen_float_buffer()
            img = 255*np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.uint8)
            self.lidar_img = img

            # close the window:
            vis.destroy_window()

            self.counter = 0

        return False

    def create_img(self, geometries):
        vis = Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.register_animation_callback(self.move_forward)
        vis.run()

        return self.lidar_img

def read_calib(calib_path):
    out = dict()
    for line in open(calib_path, 'r'):
        # print(line)
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        val = line.split(':')
        # assert len(val) == 2, 'Wrong file format, only one : per line!'
        key_name = val[0].strip()
        val = np.asarray(val[-1].strip().split(' '), dtype='f8')
        # assert len(val) in [12, 9], "Wrong file format, wrong number of numbers!"
        if len(val) == 12:
            out[key_name] = val.reshape(3, 4)
        elif len(val) == 9:
            out[key_name] = val.reshape(3, 3)
    print(out)
    return out

def load_predict(predict_filename, thresh=0.3):
    lines = [line.rstrip() for line in open(predict_filename)]
    # load as list of Object3D
    objects = []
    for line in lines:
        data = line.split(' ')
        if (float(data[15])>=0.01):
            # print(line[0])
            objects.append(Box3D(line))
    # objects = [Box3D(line) for line in lines]
    # objects = [Box3D(line[:15]) for line in lines if (float(line[15])>=thresh and line[0] != 'DontCare')]
    return objects

def load_velo_scan(velo_filename, R0_rect, Tr_velo_to_cam):
    point_cloud = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)

    # remove points that are located behind the camera:
    point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]

    point_cloud_xyz = point_cloud[:, 0:3]
    point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3]

    point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    # normalize:
    point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

    pcd = PointCloud()
    pcd.points = Vector3dVector(point_cloud_xyz_camera)
    pcd.paint_uniform_color([0.65, 0.65, 0.65])

    return pcd


def main():
    sequence = 1
    image_dir = os.path.join(DATA_DIR,'image')

    calib = read_calib(os.path.join(DATA_DIR, "calib_cam_to_cam.txt"))
    P2 = calib["P_rect_02"]
    Tr_velo_to_cam_orig = calib["P_rect_02"]
    R0_rect_orig = calib["R_02"]

    R0_rect = np.eye(4)
    R0_rect[0:3, 0:3] = R0_rect_orig

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

    list_images = sorted(os.listdir(image_dir))

    lidar_img_creator = ImgCreatorLiDAR()

    for i, image_file in enumerate(list_images):
        image_name = image_file.split('.')[0]
        image = cv2.imread(os.path.join(DATA_DIR,'image', image_file))
        # preds = load_predict(os.path.join(DATA_DIR,'prediction', image_name+'.txt'))
        pc_velo = load_velo_scan(os.path.join(DATA_DIR,'velodyne', image_name+'.bin'), R0_rect, Tr_velo_to_cam)


        img_with_input_2Dbboxes = image
        img_with_pred_bboxes = image


        # img_with_input_2Dbboxes = draw_2d_polys_no_text(img, preds)
        # img_with_pred_bboxes = draw_3d_polys(img, preds)

        img_lidar = lidar_img_creator.create_img([pc_velo])
        cv2.imshow("Test", img_lidar)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()