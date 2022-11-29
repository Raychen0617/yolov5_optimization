## Optimization API 

[optimizer/convert_compare.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/convert_compare.py): To convert a pytorch model to a tflite model and compare the difference between their outputs<br>

[optimizer/match.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py): 
- Match a NAS model to a original model 
    - fix_nasyolo, fix_nasbackbone
- Match two pytorch models with same structure, but different hyperparameters (ex: input channels, output channels)
    - match_nas, match
- Extract Yolov5's backbone from an original model 
    - extract_backbone
<br>

[optimizer/model_evaluation.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/model_evaluation.py): Evaluate the inference time, network parameters and flops of a specific model<br>

[optimizer/loss.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/loss.py): Calculating the loss between teacher model and student model<br>

[optimizer/prune.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/prune.py): Call the function prune to prune models<br>