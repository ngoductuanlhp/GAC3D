//
// Created by cao on 19-12-16.
//
#include "NvInfer.h"
#include "/usr/local/cuda-10.2/include/cuda_runtime.h"
#include "/usr/local/cuda-10.2/include/cuda_runtime_api.h"
#include "/usr/local/cuda-10.2/include/cuda.h"


#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

using namespace std;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif

namespace nvinfer1 {
    class EntroCalibrator : public nvinfer1::IInt8EntropyCalibrator {
    public:
        EntroCalibrator(int bacthSize, const std::string &imgPath, const std::string &calibTablePath);

        virtual ~EntroCalibrator();

        int getBatchSize() const override { return batchSize;  }

        bool getBatch(void *bindings[], const char *names[], int nbBindings) override;

        const void *readCalibrationCache(std::size_t &length) override;

        void writeCalibrationCache(const void *ptr, std::size_t length) override;

    private:

        int batchSize;
        size_t inputCount;
        size_t imageIndex;

        string calibTablePath;
        vector<string> imgPaths;

        float *batchData{ nullptr };
        void  *deviceInput{ nullptr };

        bool readCache;
        vector<char> calibrationCache;
        // vector<float> prepareImage(cv::Mat& img);
    };
}
