# Neural Architecture Search (NAS) 

## Goal: 
To automatically search a network architecture that leads to the best accuracy. 

## Architecture:
Blocks: residual block , inception block, bottleneck block, etc. <br>
Layers: convs, pooling, fc, etc.<br>
Hyperparameters: number of filters, size of kernel, stride, padding, etc.<br>

## Search space:
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

## YOLOv5 tutorial code: <br>
YOLOv5s backbone yaml: [yolov5sb.yaml](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolov5sb.yaml)<br>
Main: [nas.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/nas.py) <br>
Search space construction: [common.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/common.py)
```python
# models/common.py
class NASC3(nn.Module):   #L181
class NASConv(nn.Module):    #L214
```