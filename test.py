from typing import Dict, List

import tensorrt as trt
import torch
# from common_utils import loadDataset
import ctypes

class CustomProfiler(trt.IProfiler):
    def __init__(self) -> None:
        super().__init__()
        self.history: Dict[str, List[float]] = dict()

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        if layer_name in self.history:
            self.history[layer_name].append(ms)
        else:
            self.history[layer_name] = [ms]

    def dump(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_name"] + [f"iter {i + 1}" for i in range(len(next(iter(self.history.values()))))])
            for key, value in self.history.items():
                writer.writerow([key] + list(map(str, value)))

            self.history.clear()

def test_trt(engine_path, input_data):
    # Load tensor RT engine
    logger = trt.Logger()
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    input_binding_idx = engine.get_binding_index(engine.get_tensor_name(0))
    output_binding_idx = engine.get_binding_index(engine.get_tensor_name(1))

    input_shape = (1, 16, 64, 64)
    output_shape = (1, 16, 64, 64)

    # 実行時の情報を格納
    context = engine.create_execution_context()

    # デバイス側メモリの確保
    input_buffer = torch.zeros(input_shape, dtype=torch.float16, device=torch.device("cuda"))
    output_buffer = torch.zeros(output_shape, dtype=torch.float16, device=torch.device("cuda"))
    bindings = [None, None]
    bindings[input_binding_idx] = input_buffer.data_ptr()  # type: ignore
    bindings[output_binding_idx] = output_buffer.data_ptr()  # type: ignore

    # ホスト側からデバイス側に転送
    input_buffer[0:1].copy_(input_data)

    # 計算を実行
    # context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
    context.execute_v2(bindings)
    torch.cuda.current_stream().synchronize()

    output = output_buffer[0:1]
    torch.cuda.synchronize()

    return output

if __name__ == "__main__":
    ctypes.CDLL("depthwiseConvolutionPlugin/build/libdepthwiseConvolutionPlugin.so")
    torch.manual_seed(0)
    device = torch.device("cuda")
    input_data = torch.randn(1, 16, 64, 64, dtype=torch.float16).to(device)
    res_trt = test_trt("depthwiseTensorrt/dense/depthwiseTensorrt.trt", input_data)
    res_trt_with_plugin = test_trt("depthwiseWithPlugin/dense/depthwiseWithPlugin.trt", input_data)
    print("mean of relative error:", torch.mean((res_trt - res_trt_with_plugin)/ res_trt).item())
    are_equal = torch.allclose(res_trt, res_trt_with_plugin, atol=1e-1)

    if are_equal:
        print("The outputs are approximately equal.")
    else:
        print("The outputs are not equal.")