import onnx_graphsurgeon as gs
import onnx
import numpy as np

@gs.Graph.register()
def replace_with_my_kernel(self, inputs, outputs):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="DepthwiseConvolutionPlugin", inputs=inputs, outputs=outputs)


# Now we'll do the actual replacement
graph = gs.import_onnx(onnx.load("depthwiseTensorrt/depthwiseTensorrt.onnx"))

tmap = graph.tensors()

# refomat weight
a = tmap["conv1.weight"].values.reshape(1,16,3,3).astype(np.float16)
a = np.transpose(a, (0, 2, 3, 1)) #ここでchannel majorに変換することでreformat_kernelが不要になる
tmap["conv1.weight"].to_constant(values = a)

# You can figure out the input and output tensors using Netron. In our case:
# Inputs: [inp, MIN_VAL, MAX_VAL]
# Outputs: [max_out]
inputs = [tmap["input.1"], tmap["conv1.weight"]]
outputs = [tmap["3"]]

graph.replace_with_my_kernel(inputs, outputs)

# Remove the now-dangling subgraph.
graph.cleanup().toposort()

# That's it!
onnx.save(gs.export_onnx(graph), "depthwiseWithPlugin/depthwiseWithPlugin.onnx")
print("Saving the ONNX model to {}".format("depthwiseTensorrt/depthwiseTensorrt.onnx"))