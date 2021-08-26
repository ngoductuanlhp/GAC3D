import numpy as np

import cv2
import os

img = cv2.imread('/home/tuan/Downloads/test_000093/000093.jpg')
mask = cv2.imread('/home/tuan/Downloads/test_000093/Masks/000093_mask_0.png')

# print(mask.shape)
# print(mask)

out = np.zeros_like(img)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        # print(mask[i][j])
        if mask[i][j][1] == 255:
            # mask[i][j] = [255,0,0]
            out[i][j][2] = int(255 * 0.2 +img[i][j][2] * 0.8)
            out[i][j][1] = int(img[i][j][1] * 0.8)
            out[i][j][0] = int(img[i][j][0] * 0.8)
        else:
            out[i][j] = img[i][j]


# alpha = 0.7
# out = cv2.addWeighted(img, alpha, mask, 1-alpha, 0.0)
cv2.imshow("test", out)
out = out[:, 0:980,:]
cv2.imwrite('/home/tuan/93_mask.jpg', out)
# cv2.waitKey(0)