//
// Created by cao on 19-12-16.
//

#include "entro_calibrator.h"
#include <fstream>
#include <iterator>

namespace nvinfer1 {
    EntroCalibrator::EntroCalibrator(int bacthSize, const std::string &imgPath,
        const std::string &calibPath):batchSize(bacthSize),calibTablePath(calibPath),imageIndex(0){
        int inputChannel = 3;
        int inputH = 384;
        int inputW = 1280;
        inputCount = bacthSize*inputChannel*inputH*inputW;
        std::fstream f(imgPath);
        if(f.is_open()){
            std::string temp;
            while (std::getline(f,temp)) imgPaths.push_back(temp);

        }
        batchData = new float[inputCount];

        std::cout << "Num of calib images: " << imgPaths.size() << std::endl;
        CUDA_CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
    }

    EntroCalibrator::~EntroCalibrator() {
        std::cout<< "Delete calibrator " << imgPaths.size() << std::endl;
        CUDA_CHECK(cudaFree(deviceInput));
        if(batchData)
            delete[] batchData;
    }

    // int EntroCalibrator::getBatchSize() {
    //     std::cout<< "Get batch size\n";
    //     return this->batchSize;
    // }

    bool EntroCalibrator::getBatch(void **bindings, const char **names, int nbBindings){
        // std::cout << "Load image " << imageIndex << std::endl;
        if (imageIndex + batchSize > int(imgPaths.size()))
            return false;
        // load batch
        float* ptr = batchData;
        
        for (size_t j = imageIndex; j < imageIndex + batchSize; ++j)
        {
            std::string img_name = "/home/ml4u/RTM3Dv2/kitti_format/data/kitti/image/" + imgPaths[j] + ".png";
            auto img = cv::imread(img_name);
            // auto inputData = prepareImage(img);
            int channel = 3;
            int input_w = 1280;
            int input_h = 384;

            cv::Mat resized;
            cv::resize(img, resized,cv::Size(384,1280),0,0);

            cv::Mat img_float;
            resized.convertTo(img_float, CV_32FC3,1./255.);

            //HWC TO CHW
            std::vector<cv::Mat> input_channels(channel);
            cv::split(img_float, input_channels);


            float mean[]= {0.485, 0.456, 0.406};
            float std[]= {0.229, 0.224, 0.225};

            // normalize
            std::vector<float> result(input_h*input_w*channel);
            auto data = result.data();
            int channelLength = input_h * input_w;
            for (int i = 0; i < channel; ++i) {
                cv::Mat normed_channel = (input_channels[i]-mean[i])/std[i];
                memcpy(data,normed_channel.data,channelLength*sizeof(float));
                data += channelLength;
            }

            if (result.size() != inputCount)
            {
                std::cout << "InputSize error. check include/ctdetConfig.h" << std::endl;
                return false;
            }
            //assert(inputData.size() == inputCount);
            memcpy(ptr,result.data(),result.size()*sizeof(float));

            ptr += result.size();
            std::cout << "load image " << imgPaths[j] << "  " << (j+1)*100./imgPaths.size() << "%" << std::endl;
        }
        imageIndex += batchSize;
        CUDA_CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = deviceInput;
        return true;
    }

    const void* EntroCalibrator::readCalibrationCache(std::size_t &length)
    {
        calibrationCache.clear();
        std::ifstream input(calibTablePath, std::ios::binary);
        input >> std::noskipws;
        if (readCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                    std::back_inserter(calibrationCache));

        length = calibrationCache.size();
        return length ? &calibrationCache[0] : nullptr;
    }

    void EntroCalibrator::writeCalibrationCache(const void *cache, std::size_t length)
    {
        std::ofstream output(calibTablePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
}
