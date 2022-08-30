# Model Optimization

## NAS 

### Goal: 
To automatically search a network architecture that leads to the best accuracy 

### Architecture:
Blocks: residual block , inception block, bottleneck block, etc. <br>
Layers: convs, pooling, fc, etc.<br>
Hyperparameters: # of filters, size of kernel, stride, etc.<br>

### Search space:
The set containing all the possible architectures, too huge to use random search 
<br>
Example of constructing search space, for more [details](https://nni.readthedocs.io/en/stable/nas/construct_space.html)
```python
# import nni.retiarii.nn.pytorch as nn
self.layer = nn.LayerChoice([
    ops.PoolBN('max', channels, 3, stride, 1),
    ops.SepConv(channels, channels, 3, stride, 1),
    nn.Identity()
])
```

### Tutorial code: <br>
Main: [nas.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/nas.py) <br>
Search space construction: [common.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/common.py)
```python
class NASC3(nn.Module):   #L181
class NASConv(nn.Module):    #L214
```


## Pruning 

## Knowledge distillation 

