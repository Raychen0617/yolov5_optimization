# %%
#  use the code for testing latency from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import timm
import torch 
import time
import numpy as np
from nni.compression.pytorch import apply_compression_results, ModelSpeedup
import torch.nn as nn 

device = "cpu"
'''
model_path = "../2022-07-26-15-38-02-057344/best_result"
model = torch.load(model_path+"/model.pth").to(device)
dummy_input = torch.rand(10, 3, 32, 32)

from nni.compression.pytorch import ModelSpeedup
ModelSpeedup(model, torch.rand(1, 3, 32, 32), model_path+"/mask.pth").speedup_model()
'''

# Speedup the model and show the model structure after speedup.

model = timm.create_model('resnet50', pretrained=False)
model.fc  = nn.Linear(model.fc.in_features, 10)
from nni.compression.pytorch import ModelSpeedup
ModelSpeedup(model, torch.rand(1, 3, 32, 32).to(device), "../2022-07-27-13-06-54-431618/best_result/masks.pth").speedup_model()
model.load_state_dict(torch.load("../output/pruned.pth"))



from tinynn.converter import TFLiteConverter

output_path = "../output/float.tflite"
dummy_input = torch.rand(1, 3, 32, 32)
converter = TFLiteConverter(model, dummy_input, output_path)
converter.convert()


import tensorflow as tf
from tinynn.converter import TFLiteConverter
from tinynn.converter.base import GraphOptimizer

# Flag variable whether you want to compare the output tensors or all the intermediate tensors
# The suggestion is to use layerwise comparison only when the outputs don't match.
layerwise = False

# As for layerwise comparison, we need to pass `experimental_preserve_all_tensors=True`,
# which requires `tensorflow >= 2.5.0`.
tfl_interpreter_args = {'model_path': output_path}
if layerwise:
    tfl_interpreter_args['experimental_preserve_all_tensors'] = True

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(**tfl_interpreter_args)
interpreter.allocate_tensors()

# Get input and output tensors from the TFLite interpreter
input_details = interpreter.get_input_details()

if layerwise:
    details = interpreter.get_tensor_details()
else:
    details = interpreter.get_output_details()

tfl_tensor_idx_map = {t['name']: t['index'] for t in details}

# Prepare inputs for the TFLite interpreter
tfl_inputs = converter.inputs_for_tflite()

for i, t in enumerate(tfl_inputs):
    interpreter.set_tensor(input_details[i]['index'], t)

# Inference
interpreter.invoke()

# Get common nodes
torch_names = converter.tensor_names()
common_names = set(torch_names).intersection(set(tfl_tensor_idx_map))

atol = 1e-5
rtol = 1e-3

for n in torch_names:
    if n not in common_names:
        continue

    # Get outputs from the backends
    tfl_v = interpreter.get_tensor(tfl_tensor_idx_map[n])
    torch_v = converter.get_value(n)

    # Convert the PyTorch tensor to a NumPy array
    if torch_v.dtype in (torch.quint8, torch.qint8):
        torch_v = torch_v.dequantize().numpy() / torch_v.q_scale() + torch_v.q_zero_point()
    else:
        torch_v = torch_v.numpy()

    # Align shapes and dtypes of the tensors
    if torch_v.shape != tfl_v.shape:
        torch_v = np.transpose(torch_v, (0, 2, 3, 1))

    if torch_v.dtype != tfl_v.dtype:
        tfl_v = tfl_v.astype(torch_v.dtype)

    # Compare the tensors using `np.allclose`
    matches = np.allclose(tfl_v, torch_v, rtol=rtol, atol=atol)
    print(f'Output {n} value matches: {matches}')

    # Calculate absolute difference
    diff = np.abs(torch_v - tfl_v)

    diff_mean = np.mean(diff)
    diff_min = np.min(diff)
    diff_max = np.max(diff)

    abs_err_percent = np.mean((diff > atol).astype('float32')) * 100
    print(
        f'Output {n} absolute difference min,mean,max: {diff_min},{diff_mean},{diff_max} (error:'
        f' {abs_err_percent:.2f}%)'
    )

    # Calculate relative difference
    torch_v_nonzero = (torch_v != 0).astype('bool')
    if np.all(~torch_v_nonzero):
        rel_err = np.array([float('inf')] * len(torch_v))
    else:
        rel_err = diff[torch_v_nonzero] / np.abs(torch_v[torch_v_nonzero])

    rel_diff_mean = np.mean(rel_err)
    rel_diff_min = np.min(rel_err)
    rel_diff_max = np.max(rel_err)

    rel_err_percent = np.mean((rel_err > rtol).astype('float32')) * 100
    print(
        f'Output {n} relative difference min,mean,max: {rel_diff_min},{rel_diff_mean},{rel_diff_max} (error:'
        f' {rel_err_percent:.2f}%)'
    )
