# Iterative Pruning

## Preparations

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

## Finetuning

## Iteratively Prune & Finetuning
```python 
iterations = 5

for iter in range(iterations):
    backbone.to(device=device)
    backbone = prune(model=backbone, save=None, sparsity=0.1, method="L2")
    yolo = match(yolo=yolo,  pruned_yolo=backbone, save=None)
    yolo.to(device=device)
    # train & eval
    opt, yolo = train.run(data='coco128.yaml', imgsz=640, cfg='./models/yolov5s.yaml', run_model=yolo, epochs=1)
    # load yolo state dict back to backbone
    backbone.load_state_dict(yolo.state_dict(), strict=False)
```

## Full Code On Github
[iterative_pruning.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/iterative_pruning.py)
