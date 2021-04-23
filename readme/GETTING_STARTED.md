## Training
Run following command to train model with DLA-34 backbone:

   ```bash
   python3 ./src/main.py --data_dir ./kitti_format --exp_id XXX --arch dladgpac_34 --batch_size 8 --gpus 0 --num_epochs 200 --start_epoch 0 --lr 1.25e-4 --not_rand_crop --load_model XXX
   ```
   
## Results generation
Run following command for results generation:
   * *Val* set:
   ```bash
   python3 ./src/infer.py --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/pretrained/2d3d_noaug_lr_trainval_last.pth --gpus 0 --arch dladepthconv_34
   ```

   * *Test* set:
   ```bash
   python3 ./src/infer.py --demo ./kitti_test/data/kitti/test.txt --data_dir ./kitti_test --calib_dir ./kitti_test/data/kitti/calib/ --load_model ./kitti_format/pretrained/2d3d_noaug_lr_trainval_last.pth --gpus 0 --arch dladepthconv_34
   ```

## Evaluation
Run following command for evaluation:
   * single class (Car)
   ```bash
   python ./src/tools/kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti_format/data/kitti/label/ --label_split_file ./kitti_format/data/kitti/val.txt --current_class=0 --coco=False --result_path=./kitti_format/exp/results/data/
   ```

   * multi classes
   ```bash
   python ./src/tools/kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti_format/data/kitti/label/ --label_split_file ./kitti_format/data/kitti/val.txt --current_class=0,1,2 --coco=False --result_path=./kitti_format/exp/results/data/
   ```

## Visualization
Run following command for visualization.
   ~~~
   # ResNet-18 backbone
   python ./src/faster.py --vis --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_res18/model_last.pth --gpus 0 --arch res_18
   # or DLA-3D backbone
   python ./src/faster.py --vis --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_dla34/model_last.pth --gpus 0 --arch res_18
   ~~~
## All-in-one fashion
Generating results in all-in-one fashion. It simultaneously generates main center, 2D bounding box, regressed keypoints, heatmap keypoint, dim, orientation, confidence. 
      
      ~~~
        # ResNet-18 backbone
        python ./src/faster.py --demo ./kitti_format/data/kitti/val.txt --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_res18/model_last.pth --gpus 0 --arch res_18
        # or DLA-3D backbone
        python ./src/faster.py --demo ./kitti_format/data/kitti/val.txt --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_dla34/model_last.pth --gpus 0 --arch res_18
      ~~~

