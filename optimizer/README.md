# Model Optimization

## NAS 

### Goal: 
To automatically search a network architecture that leads to the best accuracy. 

### Architecture:
Blocks: residual block , inception block, bottleneck block, etc. <br>
Layers: convs, pooling, fc, etc.<br>
Hyperparameters: # of filters, size of kernel, stride, etc.<br>

### Search space:
The set containing all the possible architectures <br>
Example of constructing search space, for more details, please refer to [NNI](https://nni.readthedocs.io/en/stable/nas/construct_space.html)
```python
# import nni.retiarii.nn.pytorch as nn
self.layer = nn.LayerChoice([
    ops.PoolBN('max', channels, 3, stride, 1),
    ops.SepConv(channels, channels, 3, stride, 1),
    nn.Identity()
])
```

### YOLOV5 tutorial code: <br>
YOLO backbone yaml: [yolov5sb.yaml](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolov5sb.yaml)<br>
Main: [nas.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/nas.py) <br>
Search space construction: [common.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/common.py)
```python
class NASC3(nn.Module):   #L181
class NASConv(nn.Module):    #L214
```

## Pruning 
### Goal: 
Aim to remove unimportant channels while minimizing the reconstruction error  <br>

### Ranking algorithm: 
To rank the importance of every channel (ex: L1, L2, APOz, FPGM, etc.) <br>
Example of channel pruning, for more details, please refer to [NNI](https://nni.readthedocs.io/en/stable/tutorials/pruning_quick_start_mnist.html)
```python 
from nni.compression.pytorch.pruning import L1NormPruner
pruner = L1NormPruner(model, config_list)
# compress the model and generate the masks
_, masks = pruner.compress()
```
### YOLOV5 tutorial code: <br>
YOLO backbone yaml: [yolov5sb.yaml](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolov5sb.yaml)<br>
Main: [pruning.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/pruning.py) <br>
Pruning Algorithm: [optimizer/prune.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/prune.py)<br>

## Knowledge distillation 
### Goal 
To Improve student’s accuracy with the help of our teacher model 
<br>

## Optimization tools
[optimizer/convert_compare.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/convert_compare.py) To convert a pytorch model to a tflite model and compare the difference between their outputs<br>
[optimizer/match.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py) Match backbone (structually changed by NAS or pruning) back to a YOLO model<br>
[model_evaluation.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/model_evaluation.py): Evaluate the inference time, network parameters and flops of a specific model<br>

