# GAC3D: Improving monocular 3D object detection with ground-guide model and adaptive convolution

## Introduction
This work proposes a novel approach for 3D object detection by employing a ground plane model that utilizes geometric constraints, named GAC3D. This approach improves the results of the deep-based detector. Furthermore, we introduce a depth adaptive convolution to replace the traditional 2D convolution to deal with the divergent context of the image's feature, leading to a significant improvement in both training convergence and testing accuracy. We demonstrate our approach on the KITTI 3D Object Detection benchmark, which outperforms existing monocular methods.

The structure of this repo is as follows:
```bash
GAC3D
├── kitti_format # for training and evaluating on trainval set
├── kitti_test # for testing on official test set
├── src # source code for implementing the framework
├── trt_src # source code for deploying the framework on NVidia Jetson board
├── readme # readme files
```

## Requirements
Our framework is implemented and tested with Ubuntu 18.04, CUDA 10.2, CuDNN 7.5.1, Python 3.6, Pytorch 1.7, single NVIDIA RTX 2080.

## Dataset preparation
Download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
GAC3D
├── kitti_format
│   ├─ data
│   │   ├── kitti
│   │   |   ├── annotations 
│   │   │   ├── calib /000000.txt .....
│   │   │   ├── image(left[0-7480] right[7481-14961])
│   │   │   ├── label /000000.txt .....
|   |   |   ├── train.txt val.txt trainval.txt
```

## Installation
* For training and testing on mainframe computer: [INSTALL.md](readme/INSTALL.md)
* For testing on embedded device (we use NVidia Jetson Xavier NX): [INSTALL_JETSON.md](readme/INSTALL_JETSON.md)

## Demo
Please refer to [GETTING_STARTED.md](readme/GETTING_STARTED.md) to learn more usage about this project.

## NVidia Jetson deployment
We deployed and tested our framework on NVidia Jetson XavierNX with Jetpack 4.5.1. Please follow these steps below to run the framework on Jetson board:

* Install packages on Jetson: [INSTALL_JETSON.md](readme/INSTALL_JETSON.md)
* Build TensorRT src and additional plugins: [BUILD_TENSORRT.md](readme/BUILD_TENSORRT.md)
* Infer model on Jetson: [DEMO_JETSON.md](readme/DEMO_JETSON.md)

## Acknowledgement
Portions of the code are borrowed from:
* [CenterNet](https://github.com/xingyizhou/CenterNet)
* [RTM3D](https://github.com/Banconxuan/RTM3D)
* [DCNv2](https://github.com/jinfagang/DCNv2_latest) (Deformable Convolutions)
* [PAC](https://github.com/NVlabs/pacnet) (Pixel Adaptive Convolution)
* [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation).
<!-- * [iou3d](https://github.com/sshaoshuai/PointRCNN) -->
<!-- ## License

RTM3D and KM3D are released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from, [CenterNet](https://github.com/xingyizhou/CenterNet), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [iou3d](https://github.com/sshaoshuai/PointRCNN) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)). -->