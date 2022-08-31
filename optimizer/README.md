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
To Improve student’s accuracy with the help of our teacher model <br>
Example of Knowledge Distillation 
```python 
class SoftTarget(nn.Module):
	
	#   Distilling the Knowledge in a Neural Network: https://arxiv.org/pdf/1503.02531.pdf
        #   The only change for KD from original training is to implement a new loss function 
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss
```
Main: [training.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/training.py) Integrated in training.py, specify `--t_weights` to execute KD <br>
KD Algorithm: [optimizer/loss.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/loss.py)<br> 
<br>

## Optimization tools
[optimizer/convert_compare.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/convert_compare.py) To convert a pytorch model to a tflite model and compare the difference between their outputs<br>
[optimizer/match.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py) Match backbone (structually changed by NAS or pruning) back to a YOLO model<br>
[model_evaluation.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/model_evaluation.py): Evaluate the inference time, network parameters and flops of a specific model<br>
[optimizer/loss.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/loss.py): Calculating the loss between teacher model and student model
