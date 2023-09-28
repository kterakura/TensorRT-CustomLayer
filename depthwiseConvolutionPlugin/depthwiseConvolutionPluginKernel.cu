#include "depthwiseConvolutionPlugin.h"
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{

__global__ void dw_conv_fp16(__half* input, __half* filter, __half* output) {
    const int tid = threadIdx.x;
    const int output_x = tid/16 + blockIdx.x*16;
    const int output_y = blockIdx.y;
    const int out = output_y*16*64 + output_x*16 + tid%16;
    int filter_x;
    int filter_y;
    int fil;
    int input_x;
    int input_y;
    int in;
    half reg_out = 0;
    for(int i=0; i<9; i++){
        filter_x = i%3;
        filter_y = i/3;
        fil = i*16 + tid%16;
        input_x = output_x-1 + filter_x;
        input_y = output_y-1 + filter_y;
        in = input_x*16 + input_y*16*64 + tid%16;
        if(input_x >= 0 && input_y >= 0 && input_x < 64 && input_y < 64) reg_out += __hmul(input[in], filter[fil]);
    }
    // leaky relu
    output[out] = __hgt(reg_out, __float2half(0.0f)) ? reg_out : __float2half(0.1f)*reg_out;
    return;
}

template <typename T>
int inferenceDepthwiseConvolution_fp16(
    T* inputs, T* weights, T* outputs, cudaStream_t stream)
{
    // NCHW
    dw_conv_fp16<<<dim3(4, 64), 256, 0, stream>>>(inputs, weights, outputs);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


int DepthwiseConvolutionPlugin::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    switch (iType)
    {
    case DataType::kFLOAT:
    case DataType::kHALF:
        return inferenceDepthwiseConvolution_fp16((__half*) inputs[0], (__half*) inputs[1], (__half*) outputs[0], stream);
    case DataType::kINT8:
    case DataType::kUINT8:
    case DataType::kINT32:
    case DataType::kBOOL:
        break;
    case DataType::kFP8: PLUGIN_FAIL("FP8 not supported"); break;
    }
    return 1;
}
} // namespace plugin
} // namespace nvinfer1
