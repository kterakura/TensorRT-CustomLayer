#!/bin/bash

# declare -a str_array=("depthwiseTensorrt")
declare -a str_array=("depthwiseWithPlugin")
SO_PATH="depthwiseConvolutionPlugin/build/libdepthwiseConvolutionPlugin.so"
USE_FP16=true

for i in "${str_array[@]}"
do
   echo $i
   if [ ! -d "${i}/dense" ]; then
      mkdir -p "${i}/dense"
   fi
   if [ ! -f "${i}/dense/${i}.trt" ]; then
      if [ "$USE_FP16" = true ]; then
         /usr/src/tensorrt/bin/trtexec --verbose --precisionConstraints=obey --fp16 --layerPrecisions=*:fp16 --inputIOFormats=fp16:hwc8 --outputIOFormats=fp16:hwc8 --buildOnly --onnx=${i}/${i}.onnx --saveEngine=${i}/dense/${i}.trt --timingCacheFile=${i}/dense/timing_cache --nvtxMode=verbose --plugins=${SO_PATH} > ${i}/dense/convert.log 2>&1
      else
         /usr/src/tensorrt/bin/trtexec --verbose --inputIOFormats=fp32:hwc8 --outputIOFormats=fp32:hwc8 --buildOnly --onnx=${i}/${i}.onnx --saveEngine=${i}/dense/${i}.trt --timingCacheFile=${i}/dense/timing_cache --nvtxMode=verbose --plugins=${SO_PATH} > ${i}/dense/convert.log 2>&1
      fi
   fi
done





