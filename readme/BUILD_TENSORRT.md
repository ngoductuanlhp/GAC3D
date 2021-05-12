1. Build plugin of TensorRT:
```bash
cd trt_src/onnx-tensorrt/plugin &&\
mkdir build && cd build &&\
cmake .. &&\
make -j6
```

2. Build `onnx-tensorrt`:
```bash
cd trt_src/onnx-tensorrt &&\
mkdir build && cd build &&\
cmake .. &\
make -j6
```

3. Build DCNv2 compatible with TensorRT plugin:
Replace folder /src/lib/models/networks/DCNv2 with /trt_src/DCNv2_jetson and rename the folder as DCNv2

Build DCNv2 src:

```bash
cd src/lib/models/networks/DCNv2
sudo ./make.sh
```

