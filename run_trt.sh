#!/bin/bash
# declare -a str_array=("depthwiseTensorrt")
declare -a str_array=("depthwiseWithPlugin")
SO_PATH="depthwiseConvolutionPlugin/build/libdepthwiseConvolutionPlugin.so"


for i in "${str_array[@]}"
do
   echo $i
   if [ -f "${i}/dense/${i}.trt" ] && [ ! -f "${i}/dense/gpu_compute_time.txt" ]; then
      /usr/src/tensorrt/bin/trtexec  --verbose --noDataTransfers --useCudaGraph --separateProfileRun --useSpinWait --nvtxMode=verbose --loadEngine=${i}/dense/${i}.trt --exportTimes=${i}/dense/times.json --exportProfile=${i}/dense/profile.json --exportLayerInfo=${i}/dense/layer_info.json --timingCacheFile=${i}/dense/timing_cache --plugins=${SO_PATH} | grep "GPU Compute Time:" > ${i}/dense/gpu_compute_time.txt
   fi
done
