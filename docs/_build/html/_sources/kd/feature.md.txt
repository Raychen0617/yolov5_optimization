# Distill features 

The paper implemented in this experiement is [Distilling-Object-Detectors](https://www.notion.so/chentzj/Feature-Distill-For-Detection-Models-d9630e17e8534041a3c8feeabf517c84#9cd5ae365dc74558a2acf2a89cae70d5)<br>

Goal: Not learning the whole feature map (too complicated for the student). Learn near object anchor locations’ features. 

![](./distill_feature.png)


## Modify Model 
Add function to make center anchors (fixed type)
```python 
def make_center_anchors(anchors_wh, grid_size=80, device='cpu'):

    grid_arange = torch.arange(grid_size)
    xx, yy = torch.meshgrid(grid_arange, grid_arange)  # + 0.5  # grid center, [fmsize*fmsize,2]
    xy = torch.cat((torch.unsqueeze(xx, -1), torch.unsqueeze(yy, -1)), -1) + 0.5

    wh = torch.tensor(anchors_wh)

    xy = xy.view(grid_size, grid_size, 1, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # centor
    wh = wh.view(1, 1, 5, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # w, h
    center_anchors = torch.cat([xy, wh], dim=3).to(device)
    # cy cx w h

    """
    center_anchors[0][0]
    tensor([[ 0.5000,  0.5000,  1.3221,  1.7314],
            [ 0.5000,  0.5000,  3.1927,  4.0094],
            [ 0.5000,  0.5000,  5.0559,  8.0989],
            [ 0.5000,  0.5000,  9.4711,  4.8405],
            [ 0.5000,  0.5000, 11.2364, 10.0071]], device='cuda:0')
            
    center_anchors[0][1]
    tensor([[ 1.5000,  0.5000,  1.3221,  1.7314],
            [ 1.5000,  0.5000,  3.1927,  4.0094],
            [ 1.5000,  0.5000,  5.0559,  8.0989],
            [ 1.5000,  0.5000,  9.4711,  4.8405],
            [ 1.5000,  0.5000, 11.2364, 10.0071]], device='cuda:0')
            
    center_anchors[1][0]
    tensor([[ 0.5000,  1.5000,  1.3221,  1.7314],
            [ 0.5000,  1.5000,  3.1927,  4.0094],
            [ 0.5000,  1.5000,  5.0559,  8.0989],
            [ 0.5000,  1.5000,  9.4711,  4.8405],
            [ 0.5000,  1.5000, 11.2364, 10.0071]], device='cuda:0')
    
    pytorch view has reverse index
    """

    return center_anchors
```


Add the function to get intermediate feature to our model 
```python 
def _forward_once(self, x, profile=False, visualize=False, target=None):
    y, dt = [], []  # outputs
    cnt = 0
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if isinstance(m, Concat):
            cnt += 1
            if cnt == 2:
                feature = x
    if target is not None:
        return x, feature
    return x
```

Add the function to get imitation mask to our model 
```python 
def _get_imitation_mask(self, x, targets, iou_factor=0.5):
    """
    gt_box: (B, K, 4) [x_min, y_min, x_max, y_max]
    """
    self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                    (11.2364, 10.0071)]

    self.num_anchors = len(self.anchors)

    out_size = x.size(2)
    batch_size = x.size(0)
    device = targets.device

    mask_batch = torch.zeros([batch_size, out_size, out_size])

    if not len(targets):
        return mask_batch

    gt_boxes = [[] for i in range(batch_size)]
    for i in range(len(targets)):
        gt_boxes[int(targets[i, 0].data)] += [targets[i, 2:].clone().detach().unsqueeze(0)]

    max_num = 0
    for i in range(batch_size):
        max_num = max(max_num, len(gt_boxes[i]))
        if len(gt_boxes[i]) == 0:
            gt_boxes[i] = torch.zeros((1, 4), device=device)
        else:
            gt_boxes[i] = torch.cat(gt_boxes[i], 0)

    for i in range(batch_size):
        # print(gt_boxes[i].device)
        if max_num - gt_boxes[i].size(0):
            gt_boxes[i] = torch.cat((gt_boxes[i], torch.zeros((max_num - gt_boxes[i].size(0), 4), device=device)), 0)
        gt_boxes[i] = gt_boxes[i].unsqueeze(0)


    gt_boxes = torch.cat(gt_boxes, 0)
    gt_boxes *= out_size

    center_anchors = make_center_anchors(anchors_wh=self.anchors, grid_size=out_size, device=device)
    anchors = center_to_corner(center_anchors).view(-1, 4)  # (N, 4)

    gt_boxes = center_to_corner(gt_boxes)
    mask_batch = torch.zeros([batch_size, out_size, out_size], device=device)

    for i in range(batch_size):
        num_obj = gt_boxes[i].size(0)
        if not num_obj:
            continue

        IOU_map = find_jaccard_overlap(anchors, gt_boxes[i], 0).view(out_size, out_size, self.num_anchors, num_obj)
        max_iou, _ = IOU_map.view(-1, num_obj).max(dim=0)
        mask_img = torch.zeros([out_size, out_size], dtype=torch.int64, requires_grad=False).type_as(x)
        threshold = max_iou * iou_factor

        for k in range(num_obj):

            mask_per_gt = torch.sum(IOU_map[:, :, :, k] > threshold[k], dim=2)

            mask_img += mask_per_gt

            mask_img += mask_img
        mask_batch[i] = mask_img

    mask_batch = mask_batch.clamp(0, 1)
    return mask_batch  # (B, h, w)
```

Change the forward function of our model to fit all cases (distill or not)
```python 

def forward(self, x, augment=False, profile=False, visualize=False, target=None):
    if augment:
        return self._forward_augment(x)  # augmented inference, None
    if target != None: 
        preds, features = self._forward_once(x, profile, visualize, target)
        mask = self._get_imitation_mask(features, target).unsqueeze(1)
        return preds, features, mask
    return self._forward_once(x, profile, visualize)  # single-scale inference, train
``` 

## New Loss Calculation

Add a function that returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
```python
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU: 
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
```

Calculate the Loss Between Teacher's And Student's Imitation_mask
```python
def imitation_loss(teacher, student, mask):
    if student is None or teacher is None:
        return 0
    # print(teacher.shape, student.shape, mask.shape)
    diff = torch.pow(student - teacher, 2) * mask
    diff = diff.sum() / mask.sum() / 2

    return diff
```

Modify Computeloss Class, full code please refer to [here](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/loss.py#L156)
```python
def __call__(self, p, targets, teacher=None, student=None, mask=None):  # predictions, targets, model
    lmask = imitation_loss(teacher, student, mask) * 0.01
```


## Run the Code
```bash
python train.py --data coco.yaml --epochs 101 --weights "original_model" --ft_weights "teacher_model" 
```
