/*
 * custom plugin, TensorRT-7
 *
 */ 
#ifndef DCN_V2_HPP
#define DCN_V2_HPP

#include "NvInfer.h"

#include <thread>

#include <cassert>
#include <iostream>
#include <vector>


constexpr const char* DCN_PLUGIN_VERSION{"v2"};
constexpr const char* DCN_PLUGIN_NAME{"DCN"};

namespace SeriBuff{
    //Write values into buffer
    template <typename T>
    void write(char*& buffer, const T& val) {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    // Read values from buffer
    template <typename T>
    T read(const char*& buffer) {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }
}


/// inherited from IPluginV2Ext 
class DCNPlugin final: public nvinfer1::IPluginV2Ext {
private:
    std::string _nameSpace;
    nvinfer1::Dims _inputDims;
    nvinfer1::Dims _outputDims;
    int _kernel_size;
    int _dilation;
    int _deformable_groups;
    int _padding;
    int _stride;
    bool _initialized;
    float* _d_ones;
    float* _d_columns;
    float* _d_weight;

public:
    DCNPlugin(const void* data, size_t length) {
        using namespace SeriBuff;
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        _inputDims.nbDims = 3;
        _inputDims = nvinfer1::Dims3();
        _inputDims.d[0] = read<int>(d);
        _inputDims.d[1] = read<int>(d);
        _inputDims.d[2] = read<int>(d);
        _outputDims.nbDims = 3;
        _outputDims.d[0] = read<int>(d);
        _outputDims.d[1] = read<int>(d);
        _outputDims.d[2] = read<int>(d);
        _kernel_size = read<int>(d);
        _dilation  = read<int>(d);
        _padding   = read<int>(d);
        _stride    = read<int>(d);
        _deformable_groups = SeriBuff::read<int>(d);

        assert(d == a + length);
        //std::cout << "**** DCNPlugin Constructor2 has been called!" << std::endl;
        _initialized = false;
         initialize();
    }
    ~DCNPlugin() override {
    }
    DCNPlugin()=delete; //constructor must has arguments.
    nvinfer1::Dims const&  getInputDims(int index) const { return _inputDims; }

    /// override these methods
    int getNbOutputs() const override {return 1; }
    void terminate() override;
    void destroy() override { delete this; }
    size_t getWorkspaceSize(int) const override {return 0;}

    int initialize() override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) override {
      //std::cout << "**** getOutputDimensions called! **** id:" << this << std::endl;
      assert(index == 0);
      assert(nbInputs == 5);
      auto& input = inputDims[0];
      assert(3 == input.nbDims);  /// CHW
      assert(input.d[0] > 0 && input.d[1] > 0 && input.d[2] > 0 );

      //printf("Inputs & outputs'shape: (%d, %d, %d), (%d, %d, %d)", _inputDims.d[0], _inputDims.d[1], _inputDims.d[2], _outputDims.d[0], _outputDims.d[1], _outputDims.d[2]);
      return _outputDims;
    }

    int enqueue(int batch_size, const void* const* inputs, \
            void** outputs, void* workspace, cudaStream_t stream) override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, \
            int nbInputs) const override {assert(index == 0); 
        //return this->_dataType;
        return nvinfer1::DataType::kFLOAT;
    }

    size_t getSerializationSize() const override {
        return sizeof(int) * 11;
    }
    /// serialize the engine
    void serialize(void* buffer) const override {
        using namespace SeriBuff;
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, _inputDims.d[0]);
        write(d, _inputDims.d[1]);
        write(d, _inputDims.d[2]);
        write(d, _outputDims.d[0]);
        write(d, _outputDims.d[1]);
        write(d, _outputDims.d[2]);
        write(d, _kernel_size);
        write(d, _dilation);
        write(d, _padding);
        write(d, _stride);
        write(d, _deformable_groups);

        assert(d == a + getSerializationSize());
    }

    nvinfer1::IPluginV2Ext* clone() const override {
        return new DCNPlugin(*this);
    }
    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, \
            const nvinfer1::Dims* outputDims, int nbOutputs, \
            const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,\
            const bool* inputIsBroadcast, const bool* outputIsBroadcast, \
            nvinfer1::PluginFormat floatFormat, int maxBatchSize) override {
        bool format = supportsFormat(inputTypes[0], floatFormat);
        assert(format);
        this->_inputDims = inputDims[0];
        //this->_dataType = outputTypes[0];
        this->_outputDims = outputDims[0];
        //std::cout << "***configurePlugin called!*** id:" << this << std::endl;
    }

    /// support format: fp32/fp16 and NCHW
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        //return ((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) 
        return (type == nvinfer1::DataType::kFLOAT  
                && format == nvinfer1::PluginFormat::kNCHW);
    }
    const char* getPluginType() const override {return DCN_PLUGIN_NAME;}
    const char* getPluginVersion() const override {return DCN_PLUGIN_VERSION;}
    void setPluginNamespace(const char* libNamespace) override {_nameSpace = libNamespace;}
    //const char* getPluginNamespace() const {return _nameSpace.c_str();}
    const char* getPluginNamespace() const {return "";}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override {return false;}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator
            ) override {}
    void detachFromContext() override {}

};

/// IPluginCreator
class DCNPluginCreator : public nvinfer1::IPluginCreator {
private:
    std::string mNamespace;
    static nvinfer1::PluginFieldCollection _mFC;
    static std::vector<nvinfer1::PluginField> _mPluginAttributes;
public:
    DCNPluginCreator() { 
        //_mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT8, 1));

        //_mFC.nbFields = _mPluginAttributes.size();
        //_mFC.fields   = _mPluginAttributes.data();
    }
    ~DCNPluginCreator() {}

    const char* getPluginName() const { return DCN_PLUGIN_NAME; }

    const char* getPluginVersion() const { return DCN_PLUGIN_VERSION; }

    nvinfer1::PluginFieldCollection* getFieldNames() {
        return &_mFC;
    }

    nvinfer1::IPluginV2Ext* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
        throw std::runtime_error("Not implemented!");
        return nullptr;
    }


    nvinfer1::IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { 
        return new DCNPlugin(serialData, serialLength); 
    }

    void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

    const char* getPluginNamespace() const { return mNamespace.c_str(); }

};

/// register plugin
REGISTER_TENSORRT_PLUGIN(DCNPluginCreator);
// end of this file
#endif 
