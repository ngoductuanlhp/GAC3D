The project was implemented and tested on Ubuntu 18.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6.

1. Create conda virtual environment. 
    ```bash
    conda create -n GAC3D python=3.6 anaconda
    conda activate GAC3D
    ```
2. Install Pytorch 1.7 (with CUDA 10.2):
    ```bash
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
    ```
3. Install other python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Clone and build DCNv2 (from [DCNv2](https://github.com/jinfagang/DCNv2_latest)):
    ```bash
    cd GAC3D/src/lib/models/networks/
    git clone https://github.com/jinfagang/DCNv2_latest
    cd DCNv2
    sh make.sh
    ```
5. Compile iou3d (from [pointRCNN](https://github.com/sshaoshuai/PointRCNN)): 
    ```bash
    cd GAC3D/src/lib/utiles/iou3d
    python setup.py install
    ```

5. Compile nms (optional): 
    ```bash
    cd GAC3D/src/lib/external
    make
    ```
