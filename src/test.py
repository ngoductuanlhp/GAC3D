from lib.utils.image import flip, color_aug, get_affine_transform
import numpy as np 
import cv2

eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)


inp = cv2.imread('/home/tuan/RTM3Dv2/kitti_format/data/kitti/image/000025.png')
h, w = inp.shape[0:2]

h_ori = h
w_ori = w
h = int(h * 0.667)
# inp = (inp.astype(np.float32) / 255.)
# rng = np.random.RandomState(3)
# color_aug(rng, inp, eig_val, eig_vec)

# outp = (inp * 255.).astype(np.uint8)
# outp = cv2.flip(inp, 1)

c = np.array([w / 2., int(inp.shape[0] * 0.667)], dtype=np.float32)
s = np.array([w, h], dtype=np.int32)


# sf = 0.4
# cf = 0.1
# c[0] += inp.shape[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
# c[1] += inp.shape[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
# s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)


kp = (508, 116)

trans_input = get_affine_transform(
            c, s, 0, [w, h])

trans_input_inv = get_affine_transform(
            c, s, 0, [w, h], inv=1)

outp = cv2.warpAffine(inp, trans_input,
                             (w, h),
                             flags=cv2.INTER_LINEAR)

inp_rec = cv2.warpAffine(outp, trans_input_inv,
                             (w_ori, h_ori),
                             flags=cv2.INTER_LINEAR)



kp_i = np.array([508,116,1]).transpose()

kp_out = np.matmul(trans_input_inv, kp_i)
print(kp_out)
outp = cv2.circle(outp, kp, 2, (0,0,255), 2)
inp_rec = cv2.circle(inp_rec, (int(kp_out[0]), int(kp_out[1])), 2, (0,0,255), 2)
cv2.imshow("out", outp)
cv2.imshow("inp", inp)
cv2.imshow("inp_rec", inp_rec)
cv2.waitKey(0)
# cv2.imwrite('/home/tuan/aug.png', outp)

