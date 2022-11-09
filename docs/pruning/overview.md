
# Pruning 
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

### What's Channel Pruning

Goal: Aim to remove less important channels while minimizing the accuracy loss  
To rank the importance of channels, we have to choose a ranking algorithm (L1, L2, FPGM, etc.)

### Why Channel Pruning


### Tutorials:
[Iterative pruning on yolov5](./iterative_pruning.md) <br><br>
[Pruning on yolov5 backbone](./prune.md) 