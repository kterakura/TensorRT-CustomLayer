#!/bin/bash
# declare -a str_array=("depthwiseTensorrt")
declare -a str_array=("depthwiseWithPlugin")
SO_PATH="depthwiseConvolutionPlugin/build/libdepthwiseConvolutionPlugin.so"

# 配列の要素を順番に表示
for i in "${str_array[@]}"
do
   echo $i
   if [ -f "${i}/dense/${i}.trt" ]; then
      /opt/nvidia/nsight-compute/2023.2.2/ncu -o ${i}/dense/profile_${i} -f --set full /usr/src/tensorrt/bin/trtexec  --loadEngine=${i}/dense/${i}.trt --warmUp=0 --duration=0 --iterations=1 --useSpinWait --noDataTransfers --useCudaGraph --plugins=${SO_PATH}
   fi
done
