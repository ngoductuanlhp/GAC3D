import numpy as np

def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        ''' Default params from Kitti '''
        self.type = data[0]
        self.truncation = data[1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        # print("box 2d:", self.box2d)

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        ''' Derived params'''
        self.center_2d = [(self.xmin + self.xmax)/2, (self.ymin + self.ymax)/2]
        self.center_3d = list(self.t)
        self.center_3d[1] -= self.h/2
        
        if len(data) == 17:
            self.score_3d = data[16]
            print(self.score_3d)
        if len(data) > 17:
            self.kps = np.array(data[16:32])
            self.kps = self.kps.reshape(-1,2)
            self.kps = self.kps.transpose()

            self.ori_2d = np.array(data[32:34])
            self.ori_2d = self.ori_2d.reshape(-1,2)
            self.ori_2d = self.ori_2d.transpose()
            print(self.ori_2d)

    def get_recorrect_center(self, P_rect2cam2, center, gt=False):
        # loc = (self.t[0], self.t[1], 0)
        fx = P_rect2cam2[0][0]
        fy = P_rect2cam2[1][1]
        y0 = P_rect2cam2[1][2]

        # center2d = self.ori_2d

        # z = fy*(self.t[1] - self.h/2)/(center2d[1] - y0)
        # print(P_rect2cam2)
        # print(fy, center2d[1], self.t[1])
        if gt:
            print("GT")
        else:
            print("Pred")
        print(self.t)
        print(self.h)
        print(center)
        # print("Origin z:", self.t[2])
        # print("Recorrect z:", z)



    def in_camera_coordinate(self, is_homogenous=False):
        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3D bounding box vertices [3, 8]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]

        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return points_3d

    def get_box2d_from_3d(self, box3d):
        box3d_x = box3d[0,:]
        box3d_y = box3d[1,:]
        xmin = np.min(box3d_x)
        xmax = np.max(box3d_x)
        ymin = np.min(box3d_y)
        ymax = np.max(box3d_y)

        return np.array([[xmin,ymin],
        [xmax,ymin],
        [xmax,ymax],
        [xmin,ymax]]).astype(np.float32)

    def get_projected_3d_center(self, P_rect2cam2):
        center = np.asarray(self.t).astype(np.float32)
        center[1] -=  self.h/2
        
        center = center.reshape((3, 1))
        center = np.vstack((center, 1))
        center = (P_rect2cam2 @ center)
        
        center = center[:2,0]/center[2,0]
        return center

    def get_center3d(self):
        return np.asarray(self.center_3d).astype(np.float32).reshape((3,1))

    def get_center2d(self):
        return np.array(self.center_2d).astype(np.float32)

    def get_box2d(self):
        return np.array([[self.xmin,self.ymin],
        [self.xmax,self.ymin],
        [self.xmax,self.ymax],
        [self.xmin,self.ymax]]).astype(np.float32)
        