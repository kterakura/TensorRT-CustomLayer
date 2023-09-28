#include "depthwiseConvolutionPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
char const* const DEPTHWISECONVOLUTION_PLUGIN_VERSION{"1"};
char const* const DEPTHWISECONVOLUTION_PLUGIN_NAME{"DepthwiseConvolutionPlugin"};
// int32_t const kNUM_COORDCONV_CHANNELS = 2;
} // namespace

PluginFieldCollection DepthwiseConvolutionPluginCreator::mFC{};
std::vector<PluginField> DepthwiseConvolutionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DepthwiseConvolutionPluginCreator);

DepthwiseConvolutionPlugin::DepthwiseConvolutionPlugin() {}

DepthwiseConvolutionPlugin::DepthwiseConvolutionPlugin(
    nvinfer1::DataType iType)
    : iType(iType)
    // , iFormat(iFormat)
{
}

DepthwiseConvolutionPlugin::DepthwiseConvolutionPlugin(void const* data, size_t length)
{
    deserialize(static_cast<uint8_t const*>(data), length);
}
void DepthwiseConvolutionPlugin::deserialize(uint8_t const* data, size_t length)
{
    auto const* d{data};
    iType = read<nvinfer1::DataType>(d);
    // iFormat = read<nvinfer1::PluginFormat>(d);
    PLUGIN_VALIDATE(d == data + length);
}

int32_t DepthwiseConvolutionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DepthwiseConvolutionPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void DepthwiseConvolutionPlugin::terminate() noexcept {}

Dims DepthwiseConvolutionPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    nvinfer1::Dims dimsOutput;
    // Don't trigger null dereference since we check if inputs is nullptr above.
    PLUGIN_ASSERT(inputs[0].nbDims == 3);
    dimsOutput.nbDims = inputs[0].nbDims;
    dimsOutput.d[0] = inputs[0].d[0];
    dimsOutput.d[1] = inputs[0].d[1];
    dimsOutput.d[2] = inputs[0].d[2];
    return dimsOutput;
}

size_t DepthwiseConvolutionPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return 0;
}

size_t DepthwiseConvolutionPlugin::getSerializationSize() const noexcept
{
    return sizeof(nvinfer1::DataType);
}

void DepthwiseConvolutionPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iType);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}
void DepthwiseConvolutionPlugin::configurePlugin(
    PluginTensorDesc const* in, int32_t nbInputs, const PluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    iType = in[0].type;
}

bool DepthwiseConvolutionPlugin::supportsFormatCombination(
    int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept
{   
    switch (pos)
    {
    case 1:
        return (inOut[pos].type == DataType::kHALF) && (inOut[pos].format == PluginFormat::kLINEAR);
    
    default:
        return (inOut[pos].type == DataType::kHALF) && (inOut[pos].format == PluginFormat::kHWC8);
    }
    return 1;
}

char const* DepthwiseConvolutionPlugin::getPluginType() const noexcept
{
    return DEPTHWISECONVOLUTION_PLUGIN_NAME;
}

char const* DepthwiseConvolutionPlugin::getPluginVersion() const noexcept
{
    return DEPTHWISECONVOLUTION_PLUGIN_VERSION;
}

void DepthwiseConvolutionPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2IOExt* DepthwiseConvolutionPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new DepthwiseConvolutionPlugin(iType);
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DepthwiseConvolutionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* DepthwiseConvolutionPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

nvinfer1::DataType DepthwiseConvolutionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

bool DepthwiseConvolutionPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

bool DepthwiseConvolutionPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Plugin creator
DepthwiseConvolutionPluginCreator::DepthwiseConvolutionPluginCreator()
{
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dilations"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("group"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_shape"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("pads"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("strides"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* DepthwiseConvolutionPluginCreator::getPluginName() const noexcept
{
    return DEPTHWISECONVOLUTION_PLUGIN_NAME;
}

char const* DepthwiseConvolutionPluginCreator::getPluginVersion() const noexcept
{
    return DEPTHWISECONVOLUTION_PLUGIN_VERSION;
}

PluginFieldCollection const* DepthwiseConvolutionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2IOExt* DepthwiseConvolutionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        DepthwiseConvolutionPlugin* plugin = new DepthwiseConvolutionPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2IOExt* DepthwiseConvolutionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        DepthwiseConvolutionPlugin* plugin = new DepthwiseConvolutionPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
