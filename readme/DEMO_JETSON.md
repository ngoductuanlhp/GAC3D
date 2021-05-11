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
3. Run the RTM3D(GRM) with a pretrained model (e.g. ResNet-18train.pth) and kitti camera data as follows:
    ~~~
        cd km3d
        python ./src/demo.py --vis --demo ./demo_kitti_format/data/kitti/image --calib_dir ./demo_kitti_format/data/kitti/calib/ --load_model ./demo_kitti_format/exp/KM3D/pretrained.pth --gpus 0 --arch res_18
    ~~~
4. Run the KM3D with a pretrained model (e.g. ResNet-18train.pth) and custom camera data as follows:
    ~~~
        cd km3d
        python ./src/demo.py --vis --demo ~/your image folder --calib_dir ~/your calib folder/ --load_model ~/pretrained.pth --gpus 0 --arch res_18 or dla_34
    ~~~
5. Run the RTM3D(GRM) with a pretrained model (e.g. ResNet-18train.pth) and custom camera data as follows:
    ~~~
        cd km3d
        python ./src/demo.py --vis --demo ~/your image folder --calib_dir ~/your calib folder/ --load_model ~/pretrained.pth --gpus 0 --arch res_18 or dla_34
    ~~~

6. Export ONNX file: 
    ```bash
    python3 ./src/get_onnx.py --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model /home/ml4u/RTM3D_weights/res18_gac_base_200.pth --gpus 0 --arch resfake_18
    ```

7. Convert ONNX to TRT:
    ```bash
    ./build/onnx2trt /home/ml4u/RTM3D_weights/model_res18_3.onnx -d 8 -o /home/ml4u/RTM3D_weights/model_res18_3.trt -c /home/ml4u/RTM3Dv2/kitti_format/data/kitti/calib_int8_2.txt -b 1
    ```

    or 
    ```bash
    ./build/onnx2trt /home/ml4u/RTM3D_weights/model_res18_3.onnx -d 16 -o /home/ml4u/RTM3D_weights/model_res18_3.trt -b 1
    ```

8. Infer Jetson:
    Normal
    ```bash
    python3 ./trt_src/infer.py
    ```
    VIdeo:
    ```bash
    python3 ./trt_src/infer.py --video --demo 2011_09_26_drive_0009_sync
    ```
