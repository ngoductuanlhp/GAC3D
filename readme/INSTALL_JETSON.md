1. Install Python package

* Pillow:
```
sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev

pip3 install pillow
```

* numba + LLVM (**)
    * Install LLVM:
    ```bash
    sudo apt-get install llvm-10 llvm-10-dev
    ```

    * Install llvmlite:
    ```bash
    sudo pip3 install llvmlite 

    cd /usr/bin

    sudo ln -s llvm-config-10 llvm-config
    ```

    * Install numba:
    ```bash
    # temporaly move tbb for hiding from numba 
    sudo mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.bak

    sudo pip3 install numba

    # move back
    sudo mv /usr/include/tbb/tbb.bak /usr/include/tbb/tbb.h
    ```

* torchvision
```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch <version> https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
sudo python setup.py install     # use python3 if installing for Python 3.6
cd ../  # attempting to load torchvision from build dir will result in import error
pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6
```

* CMake (need version > 3.10 to build MAGMA)
```bash
# could use newer version
wget http://www.cmake.org/files/v3.13/cmake-3.13.0.tar.gz

tar xpvf cmake-3.13.0.tar.gz cmake-3.13.0/

cd cmake-3.13.0/

./bootstrap --system-curl

make -j6 # for nano: -j4

echo 'export PATH=$HOME/cmake-3.13.0/bin/:$PATH' >> ~/.bashrc

source ~/.bashrc
```

* MAGMA and pytorch (*****)

Follow step in repo [Install Fastai V2 on an Nvidia Jetson Nano running Jetpack 4.4]:https://github.com/streicherlouw/fastai2_jetson_nano with a little bit modification

