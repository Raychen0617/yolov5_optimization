# Iterative Pruning

## Pruning

### Pruning Function 

Full code: [prune.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/prune.py)

Change the model to make it export-friendly
```python
for k, m in model.named_modules(): 
    if isinstance(m, Conv): # assign export-friendly activations
        if isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
```

Setup model configs
```python
imgsz = (64, 64)
imgsz *= 2 if len(imgsz) == 1 else 1 # expand
gs = 32 # grid size (max stride)
imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection
```

Setup pruning configs. The following cfg_list means all layers whose type is Conv2d will be pruned. The final sparsity ratio for each layer is defined by variable sparsity. For more cfg_list settings, please refer to [compression config specification](https://nni.readthedocs.io/en/latest/compression/compression_config_list.html)
```python
cfg_list = [{
    'sparsity_per_layer': sparsity,
    'op_types': ['Conv2d'],
}]
```

There are many pruners supported by NNI, for more information, please refer to [NNI](https://nni.readthedocs.io/en/latest/compression/pruner.html)
```python
if method == "L1":
    pruner = L1NormPruner(model, cfg_list)
elif method == "L2":
    pruner = L2NormPruner(model, cfg_list)
elif method == "FPGM":
    pruner = FPGMPruner(model, cfg_list)
else:
    print("Method is not supported !!! (prune.py)")
    return 
```

Generate masks for each pruned layers
```python
_, masks = pruner.compress()
pruner.show_pruned_weights()
pruner._unwrap_model()
```

Masks can be used to check model performance of a specific pruning (or sparsity), but there is no real speedup. Therefore, after generating the masks, we have to replace our layers with smaller layers without masks for real speedup. 

```python
m_speedup = ModelSpeedup(model, im, masks_file=masks)
m_speedup.speedup_model()
```

### Match Function 

Extracting Yolo's backbon (full code [match.py/extract_backbone](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py#L164))
```python
def extract_backbone(backbone, yolo, backbone_layer=9):
```

Map backbone structure to Yolo (full code [match.py/match](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py#L195))

```python
def match(yolo, pruned_yolo, save):
```


### Pruning Main Code 

Load Yolo Model 
```python
yolo = torch.load(args.yolo)
```

Extract Yolo model's backbone and prune the backbone 
```python
backbone = extract_backbone(BACKBONE(cfg=ori_backbone_yaml, nc=200).backbone, yolo)
backbone = prune(model=backbone, save=None, sparsity=float(args.sparsity), method="L2")
```

Match the new backbone structure to our Yolo 
```python
yolo = match(yolo=yolo.float(),  pruned_yolo=backbone.float(), save=None)
```

## Iteratively Prune & Finetuning
```bash
# Pruning 
$ python iterative_pruning.py --yolo "./checkpoint/multi-trail_yolov5s.pt" --save_path "./iterative_pruning/yolo.pt" --sparsity 0.1 

# Finetuning
$ python train.py --data coco.yaml --weights "./iterative_pruning/yolo.pt" --img 640  --epochs 5
```

## Automatic Execution 
```bash
$ bash iterative_pruning.sh 
```

## Full Code On Github
[iterative_pruning.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/iterative_pruning.py)

