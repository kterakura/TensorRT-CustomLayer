#ifndef TRT_DEPTHWISECONVOLUTION_PLUGIN_H
#define TRT_DEPTHWISECONVOLUTION_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

#define SIZE (64)
#define CH (16)

namespace nvinfer1
{
namespace plugin
{

class DepthwiseConvolutionPlugin : public IPluginV2IOExt
{
public:
    DepthwiseConvolutionPlugin();

    DepthwiseConvolutionPlugin(DataType iType);
    
    DepthwiseConvolutionPlugin(void const* data, size_t length);

    ~DepthwiseConvolutionPlugin() override = default;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;
    
    void configurePlugin(
        PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept override;
    
    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs,
        int32_t nbOutputs) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2IOExt* clone() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

private:
    void deserialize(uint8_t const* data, size_t length);
    DataType iType{};
    char const* mPluginNamespace{};
};

class DepthwiseConvolutionPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    DepthwiseConvolutionPluginCreator();

    ~DepthwiseConvolutionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2IOExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2IOExt* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_DEPTHWISECONVOLUTION_PLUGIN_H
