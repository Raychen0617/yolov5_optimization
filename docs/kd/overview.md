
# Knowledge distillation (KD)
### Goal:
To Improve student’s accuracy with the help of a teacher model <br>
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
### YOLOv5 tutorial code: <br>
Main: [training.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/training.py) Integrated in training.py, specify `--t_weights` to execute KD <br>
KD Algorithm: [optimizer/loss.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/loss.py)<br> 
<br>
