import tensorrt as trt
import sys
import os

ONNX_NAME = '/home/ml4u/RTM3D_weights/model_res18_1.onnx'
TRT_NAME = '/home/ml4u/RTM3D_weights/model_res18_1.trt'

def get_engine(onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False, save_engine=True,max_batch_size=1):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        logger = trt.Logger(trt.Logger.VERBOSE)
        EXPLICIT_BATCH = []
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_mode  # Default: False

            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    return build_engine(max_batch_size, save_engine)


if __name__ == '__main__':
    get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME)