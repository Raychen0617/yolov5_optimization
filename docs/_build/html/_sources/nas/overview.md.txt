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

The search space visualization in this experiment
![](./NAS.jpg)


For more details, please refer to [NNI](https://nni.readthedocs.io/en/stable/nas/construct_space.html)

## Oneshot vs Multi-trail NAS 
- Multi-trail NAS: In Multi-trail NAS, users need model evaluator to evaluate the performance of each sampled model, and need an exploration strategy to sample models from a defined model space. Multi-trail mechanism is easy to understand and implement, therefore, implementing new functions (ex: efficiency limits) is much easier in Multi-trial. However, Multi-trail also needs more  traininig time than Oneshot. 
<br><br>
- Oneshot NAS: One-shot NAS algorithms leverage weight sharing among models in neural architecture search space to train a supernet, and use this supernet to guide the selection of better models. This type of algorihtms greatly reduces computational resource compared to independently training each model from scratch (Multi-trial NAS). The following figure shows how Oneshot NAS trains a supernet. 
![](./oneshot.png)

## Pros and Cons 
Pros:
- NAS can significantly increase the accuracy of models.
- Multi-trail NAS can even add time limits to ensure both accuracy and efficiency of models. 

Cons: 
- The training time for NAS is extremely long (especially Multi-trail NAS).

## Code Tutorials

[One shot nas on yolov5 backbone](./oneshot.md) <br><br>
[Multi-trial nas on yolov5](./multi-trial.md) 
