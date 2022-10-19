# Neural Architecture Search (NAS) 

## Goal 
To automatically search a network architecture that leads to the best accuracy. 

## Architecture Options
Blocks: residual block , inception block, bottleneck block, etc. <br>
Layers: convs, pooling, fc, etc.<br>
Hyperparameters: number of filters, size of kernel, stride, padding, etc.<br>

## Search space
The set containing all the possible architectures <br><br>
Example of constructing search space 

> **Warning**: Need to import nni.retiarii.nn.pytorch as nn to run the following code

```python

self.layer = nn.LayerChoice([
    ops.PoolBN('max', channels, 3, stride, 1),
    ops.SepConv(channels, channels, 3, stride, 1),
    nn.Identity()
])
```
For more details, please refer to [NNI](https://nni.readthedocs.io/en/stable/nas/construct_space.html)

## Oneshot vs Multi-trail NAS 


## Tutorials

[One shot nas on yolov5 backbone](./oneshot.md) <br><br>
[Multi-trial nas on yolov5](./multi-trial.md) 
