# Distill logits
Use teacher’s output distribution as true label <br>
Add temperature in teacher’s output to make relations more apparent <br>
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

## Distill 

Full code [optimizer/loss.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/loss.py)

### Setup 

Initializations 

```python
t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
h = model.hyp  # hyperparameters
```

Choose the loss function type 
```python
 DboxLoss = nn.MSELoss(reduction="none")
    if dist_loss == "l2":
        DclsLoss = nn.MSELoss(reduction="none")
    elif dist_loss == "kl":
        DclsLoss = nn.KLDivLoss(reduction="none")
    else:
        DclsLoss = nn.BCEWithLogitsLoss(reduction="none")
    DobjLoss = nn.MSELoss(reduction="none")
```
Iterative throught the predictions 
```python
for i, pi in enumerate(p):  # layer index, layer predictions
	t_pi = t_p[i]
	t_obj_scale = t_pi[..., 4].sigmoid()
```

Calculate Bbox errors
```python
        # BBox
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        if not reg_norm:
            t_lbox += torch.mean(DboxLoss(pi[..., :4],
                                          t_pi[..., :4]) * b_obj_scale)
        else:
            wh_norm_scale = reg_norm[i].unsqueeze(
                0).unsqueeze(-2).unsqueeze(-2)
            t_lbox += torch.mean(DboxLoss(pi[..., :2].sigmoid(),
                                          t_pi[..., :2].sigmoid()) * b_obj_scale)
            t_lbox += torch.mean(DboxLoss(pi[..., 2:4].sigmoid(),
                                          t_pi[..., 2:4].sigmoid() * wh_norm_scale) * b_obj_scale)
```

Calculate Classes errors 
```python
# Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1,
                                                           1, 1, 1, model.nc)
            if dist_loss == "kl":
                kl_loss = DclsLoss(F.log_softmax(pi[..., 5:]/T, dim=-1),
                                   F.softmax(t_pi[..., 5:]/T, dim=-1)) * (T * T)
                t_lcls += torch.mean(kl_loss * c_obj_scale)
            else:
                t_lcls += torch.mean(DclsLoss(pi[..., 5:],
                                              t_pi[..., 5:]) * c_obj_scale)
```

Scale the loss with distillation ratio 
```python
    t_lbox *= h['box'] * distill_ratio
    t_lobj *= h['obj'] * distill_ratio
    t_lcls *= h['cls'] * distill_ratio
```

## Run the code
```bash
$ python train.py --data coco.yaml --epochs 101 --weights "./checkpoint/enasv2_L1_yolov5s.pt" --t_weights "./checkpoint/yolov5m.pt" 
```

## Full Code On Github
[train.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/train.py)
