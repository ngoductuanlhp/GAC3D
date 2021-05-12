1. Go to RTM3D directory

2. Run the KM3D with a pretrained model (e.g. ResNet-18train.pth) and kitti camera data as follows:
    * ResNet18
    ~~~
        python3 ./src/faster.py --data_dir ./kitti_format --demo ./kitti_format/data/kitti/val.txt --calib_dir ./kitti_format/data/kitti/calib/ --load_model /home/ml4u/RTM3D_weights/model_res18_2.pth --gpus 0 --arch res_18
    ~~~

    * DLA34
    ~~~
        python3 ./src/faster.py --data_dir ./kitti_format --demo ./kitti_format/data/kitti/val.txt --calib_dir ./kitti_format/data/kitti/calib/ --load_model /home/ml4u/RTM3D_weights/dla_fake.pth --gpus 0 --arch dlafake_34

    ~~~

6. Export ONNX file: 
    ```bash
    python3 ./src/get_onnx.py --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model /home/ml4u/RTM3D_weights/res18_gac_base_200.pth --gpus 0 --arch resjs_18
    ```

    or 
    ```bash
    python3 ./src/get_onnx.py --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model /home/ml4u/RTM3D_weights/dla34_last.pth --gpus 0 --arch dlajs_34
    ```

7. Convert ONNX to TRT:
    ```bash
    cd trt_src/onnx-tensorrt
    ```

    ```bash
    ./build/onnx2trt /home/ml4u/RTM3D_weights/model_res18_3.onnx -d 8 -o /home/ml4u/RTM3D_weights/model_res18_3.trt -c /home/ml4u/RTM3Dv2/kitti_format/data/kitti/calib_int8_2.txt -b 1
    ```

    or 
    ```bash
    ./build/onnx2trt /home/ml4u/RTM3D_weights/dla34_last.onnx -d 16 -o /home/ml4u/RTM3D_weights/dla34_last.trt -b 1
    ```

    or 
    ```bash
    ./build/onnx2trt /home/ml4u/RTM3D_weights/model_res18_3.onnx -d 16 -o /home/ml4u/RTM3D_weights/model_res18_3.trt -b 1
    ```

8. Infer using TensorRT:
    Normal
    ```bash
    python3 ./trt_src/infer.py
    ```
    Video:
    ```bash
    python3 ./trt_src/infer.py --video --demo 2011_09_26_drive_0009_sync
    ```
