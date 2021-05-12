import cv2
import os
import numpy as np

dir_path = '/home/ml4u/GAC3D/kitti_format/data/kitti/sequences/2011_09_26_drive_0009_sync/image'

files = os.listdir(dir_path)
h, w = 0, 0
for i, f in enumerate(files):
    img_path = os.path.join(dir_path, f)
    img = cv2.imread(img_path)
    if i == 0:
        h, w = img.shape[0:2]
    else:
        if h != img.shape[0] or w != img.shape[1]:
            print("Abnormal")

    print(img.shape)
#     break

# calib_file = open(dir_path, 'r')
# for i, line in enumerate(calib_file):
#     print(line)
#     # parts = line.split(':')
#     if line.split(':')[0] == 'P_rect_02':
#         calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
#         calib = calib.reshape(3, 4)
#         break
# print(calib)
    # if i == 2:
    #     calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
    #     calib = calib.reshape(3, 4)