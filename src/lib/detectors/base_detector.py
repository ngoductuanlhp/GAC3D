from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
import os

def gen_ground(P2_prime, h, w, P0 = None):
  baseline = 0.54
  relative_elevation = 1.65
  # image = torch.from_numpy(image)
  # P2 = torch.from_numpy(P2)
  # h, w = image.shape[0], image.shape[1]
  P2 = P2_prime

  fy = P2[1:2, 1:2]  # [B, 1, 1]
  cy = P2[1:2, 2:3]  # [B, 1, 1]
  Ty = P2[1:2, 3:4]  # [B, 1, 1]

  x_range = torch.arange(w)
  y_range = torch.arange(570)
  y_range = torch.clamp(y_range, int(cy) + 1)
  _, yy_grid = torch.meshgrid(x_range, y_range)
  yy_grid = torch.transpose(yy_grid, 0, 1)
  # print(yy_grid, fy, Ty)
  # disparity = fy * baseline * \
  #     (yy_grid - cy) / (torch.abs(fy * relative_elevation + Ty) + 1e-10)
  # disparity = F.relu(disparity)

  # if P0 is not None:
  #   # Ty = (P2[1:2, 3:4] - P0[1:2, 3:4]) / P2[1:2, 1:2]
  #   Ty = 0

  z = (fy * relative_elevation + Ty) / (yy_grid - cy + 1e-10)
  # print(z)
  # print((yy_grid - cy + 1e-10))
  z = torch.clamp(z, 0, 75)
  z = z[:, 0]
  # print(Ty, fy * relative_elevation + Ty)
  return z

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
    self.image_path=' '
    const = torch.Tensor(
      [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
       [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
    self.const = const.unsqueeze(0).unsqueeze(0)
    self.const=self.const.to(self.opt.device)

    self.depth_mean = np.array([30.83664619525601], np.float32).reshape(1,1,1)  # DISP
    self.depth_std = np.array([19.992999492848206],  np.float32).reshape(1,1,1)
    # self.depth_std = np.array([4],  np.float32).reshape(1,1,1)

  def preprocess_depth(self, depth):
      n = 40
      delta = 2 * 80 / (n * ( n + 1))
      depth = 1 + 8 * (depth)/delta
      depth = -0.5 + 0.5 * np.sqrt(depth) # 0 -> 40
      depth = depth / 40 # 0 -> 1
      return depth

  def pre_process(self, image, depth, scale, meta=None):
      height, width = image.shape[0:2]
      new_height = int(height * scale)
      new_width  = int(width * scale)
      if self.opt.fix_res:
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
      else:
        inp_height = (new_height | self.opt.pad) + 1
        inp_width = (new_width | self.opt.pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
      resized_image = cv2.resize(image, (new_width, new_height))
      inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
      inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

      images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
      if self.opt.flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
      images = torch.from_numpy(images)

      resized_depth = cv2.resize(depth, (new_width, new_height))

      # FIXME test depth
      # print(resized_depth.shape)
      # dummy_depth = np.random.randint(0, 10000, size = (new_height, new_width)).astype(np.uint16)
      # resized_depth = dummy_depth
      # print(resized_depth)
      # dummy_depth = np.ones_like(resized_depth) * 10 * 256
      # s = resized_depth.shape
      # resized_depth = np.random.randn(new_width, new_height, 1)
      # resized_depth = dummy_depth

      # resized_depth = np.arange(new_width * new_height).reshape(new_height,new_width)
      # resized_depth = np.clip(resized_depth, 0, 255 * 100)
      # print(resized_depth.shape)
      inp_depth = cv2.warpAffine(
        resized_depth, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)

      inp_depth = inp_depth.astype(np.float32) / 256.0
      # NOTE test new depth preproc
      # inp_depth = self.preprocess_depth(inp_depth)
      inp_depth = inp_depth[:, :, np.newaxis]
      inp_depth = (inp_depth - self.depth_mean) / self.depth_std
      # print(np.max(inp_depth), np.min(inp_depth))
      # inp_depth = inp_depth * 10000
      depths = inp_depth.transpose(2, 0, 1).reshape(1, 1, inp_height, inp_width)
      depths = torch.from_numpy(depths)
      meta = {'c': c, 's': s,
              'out_height': inp_height // self.opt.down_ratio,
              'out_width': inp_width // self.opt.down_ratio}
      trans_output_inv = get_affine_transform(c, s, 0, [meta['out_width'], meta['out_height']],inv=1)
      trans_output_inv = torch.from_numpy(trans_output_inv)
      trans_output_inv=trans_output_inv.unsqueeze(0)
      meta['trans_output_inv']=trans_output_inv
      return images, depths, meta

  def process(self, images,meta, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results, calib):
   raise NotImplementedError

  def read_clib(self,calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
      if i == 2:
        calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        calib = calib.reshape(3, 4)
        return calib

  def read_clib0(self, calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 0:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib

  def run(self, image_or_path_or_tensor, depth_name, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''):
      self.image_path=image_or_path_or_tensor
      image = cv2.imread(image_or_path_or_tensor)
      depth = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)
      calib_path=os.path.join(self.opt.calib_dir,image_or_path_or_tensor[-10:-3]+'txt')
      calib_numpy=self.read_clib(calib_path)
      # calib_numpy0=self.read_clib0(calib_path)
      h, w = image.shape[0:2]
      ground_plane = gen_ground(calib_numpy, h, w).to(self.opt.device)
      
      calib=torch.from_numpy(calib_numpy).unsqueeze(0).to(self.opt.device)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, depths, meta = self.pre_process(image, depth, scale, meta)
        meta['trans_output_inv']=meta['trans_output_inv'].to(self.opt.device)
      else:
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      meta['calib']=calib
      meta['ground_plane'] = ground_plane
      images = images.to(self.opt.device)
      depths = depths.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, forward_time = self.process(images,depths, meta,return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results, calib_numpy)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}