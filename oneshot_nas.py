
from ast import Nonlocal
import numbers
import torch
import torch.nn.functional as F
from utils.dataloaders import create_tinyimagenet
from models.yolo import NASBACKBONE, Model
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
import json 
from optimizer.match import match_nas
from optimizer.prune import prune
from nni.retiarii import fixed_arch
import yaml
import os



device = torch.device("cuda:0")

########################    USP    ######################################
save_model_path = "./checkpoint/random_L2_yolov5s.pt"
yolo_yaml= "./models/yolov5s.yaml"
save_json_path = "./output/Random_yolov5s.json"
nas_backbone_yaml = "./models/yolov5sb_nas.yaml"
nas_traing = False
nas_full_model_weight = "./runs/train/exp2/weights/best.pt"
# Pruning configs
to_prune = True
sparsity = 0.3439
method = "L2"
save_pruned_backbone = None
########################    USP    ######################################

if nas_traing:
        
    model_space = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)
    train_dataset, test_dataset, train_loader, test_loader = create_tinyimagenet(batchsize=1024)


    evaluator = pl.Classification(
        # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
        # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
        train_dataloaders=pl.DataLoader(train_dataset, batch_size=512, num_workers=10),
        val_dataloaders=pl.DataLoader(test_dataset, batch_size=512, num_workers=10),
        # Other keyword arguments passed to pytorch_lightning.Trainer.
        max_epochs=1,
        gpus=1,
    )

    ########################    NAS algorithm   ######################################
    #exploration_strategy = strategy.ENAS(reward_metric_name='val_acc')
    exploration_strategy = strategy.DARTS()

    from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
    exp = RetiariiExperiment(model_space, evaluator, [], exploration_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'darts'
    exp_config.execution_engine = 'oneshot'


    # The following configurations are useful to control how many trials to run at most / at the same time.
    exp_config.max_trial_number = 4  # spawn 4 trials at most
    exp_config.trial_concurrency = 2  # will run two trials concurrently
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True


    # Launch the experiment
    exp.run(exp_config, 8081)

    for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict)

    with open(save_json_path, 'w') as fp:
        json.dump(model_dict, fp)


with fixed_arch(save_json_path):
    nasbackbone = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)

from optimizer.match import fix_nasbackbone, match
if to_prune:
    nasbackbone.backbone.load_state_dict(torch.load(nas_full_model_weight)['model'].state_dict(), strict=False)
    nasbackbone = prune(model=nasbackbone, save=save_pruned_backbone, sparsity=sparsity, method=method)
    
    
    from models.yolo import BACKBONE
    ori_backbone_yaml = './models/yolov5sb.yaml'
    backbone = BACKBONE(cfg=ori_backbone_yaml, nc=200).to(device=device).backbone
    backbone = fix_nasbackbone(backbone, nasbackbone.backbone)


yolo = Model(yolo_yaml).to(device=device) 
#match_nas(yolo, backbone, save_model_path)
yolo = match(yolo=yolo,  pruned_yolo=backbone, save=None)
yolo.to(device).float()
torch.save(yolo, save_model_path)
print("Success, json file is saved at ", save_json_path,"    pt file is saved at", save_model_path)
print("You can train the model by runining     python train.py --weights ", save_model_path, " --data coco.yaml --epochs 101")


# Detection oneshot NAS failed w_w 
'''

from nni.retiarii.evaluator.pytorch import ClassificationModule
class DetectionModule(ClassificationModule):
    def __init__(
        self,
        epochs: int = 1,
        hyp: dict = {}, 
        nbs: int = 64,
        batch_size: int = 32,
    ):
        # Training length will be used in LR scheduler
        super().__init__(export_onnx=False)
        self.hyp = hyp
        self.weight_decay = self.hyp['weight_decay']
        self.learning_rate = self.hyp['lr0']
        self.epochs = epochs
        self.nbs = nbs 
        self.batch_size = batch_size
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        #self.automatic_optimization = True
        
    @property
    def automatic_optimization(self) -> bool:
        return True
    def configure_optimizers(self):
        
        """Customized optimizer with momentum, as well as a scheduler."""
        # Optimizer
        accumulate = max(round(self.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.batch_size * accumulate / self.nbs  # scale weight_decay
        from utils.torch_utils import smart_optimizer
        #optimizer = smart_optimizer(self.model, 'SGD', self.hyp['lr0'], self.hyp['momentum'], self.hyp['weight_decay'])
        from torch.optim import SGD
        optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.hyp["momentum"])
        
        # Scheduler
        lf = lambda x: (1 - x / self.epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
        
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
    
        imgs, targets, _ , _ = batch
        print(imgs.shape)
        imgs = imgs.float() / 255
        from utils.loss import NASComputeLoss
        compute_loss = NASComputeLoss(model=self, h=self.hyp)
        pred = self(imgs)
        loss, loss_items = compute_loss(pred, targets)
    
        #print(loss)
        # Backward
        
        optimizer = self.optimizers()[0]
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        self.lr_schedulers().step()
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        import numpy as np
        nc = 5  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        from utils.metrics import ConfusionMatrix
        names = dict(enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names))
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loss = torch.zeros(3, device=device)
        jdict, stats, ap, ap_class = [], [], [], []
        from utils.metrics import box_iou,  ap_per_class
        im, targets, paths, shapes = batch
        im = im.float()
        nb, _, height, width = im.shape  # batch size, channels, height, width
        out, train_out = self.model(im)
        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            from pathlib import Path
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue
            # Predictions
            predn = pred.clone()
            from utils.general import scale_coords, xywh2xyxy
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            def process_batch(detections, labels, iouv):
                import numpy as np
                correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
                iou = box_iou(labels[:, 1:], detections[:, :4])
                correct_class = labels[:, 0:1] == detections[:, 5]
                for i in range(len(iouv)):
                    x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            # matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), i] = True
                return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            # Compute metrics
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="./", names=names)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
            maps = np.zeros(nc) + map
            for i, c in enumerate(ap_class):
                maps[c] = ap[i]
            self.log('val_loss', map, prog_bar=True)
            return map50
import io
import json
import tempfile
import contextlib
from pycocotools.cocoeval import COCOeval
def COCOEvaluator(json_list, val_dataset):
    # detections: (x1, y1, x2, y2, obj_conf, class_conf, class)
    cocoGt = val_dataset.coco
    # pycocotools box format: (x1, y1, w, h)
    annType = ["segm", "bbox", "keypoints"]
    if len(json_list) > 0:
        _, tmp = tempfile.mkstemp()
        json.dump(json_list, open(tmp, "w"), skipkeys=True, ensure_ascii=True)
        cocoDt = cocoGt.loadRes(tmp)
        coco_pred = {"images": [], "categories": []}
        for (k, v) in cocoGt.imgs.items():
            coco_pred["images"].append(v)
        for (k, v) in cocoGt.cats.items():
            coco_pred["categories"].append(v)
        coco_pred["annotations"] = json_list
        # json.dump(coco_pred, open("./COCO_val.json", "w"))
        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info = redirect_string.getvalue()
        return cocoEval.stats[0], cocoEval.stats[1], info
    else:
        return 0.0, 0.0, "No detection!"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Parameters
hyp = 'data/hyps/hyp.scratch-low.yaml'
if isinstance(hyp, str):
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f) 
    
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
imgsz = 640
batch_size = 8
single_cls = False
from utils.general import colorstr
train_path = "/home/raytjchen/Desktop/code/datasets/coco128/images/train2017"
gs = 32
nbs = 64  # nominal batch size
epochs = 4 # how many epochs to train for a single choice 
# Optimizer
accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
print("**********************************************", accumulate)
hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
# Create Trainloader
from utils.dataloaders import create_dataloader
train_loader, dataset = create_dataloader(train_path,
                                            imgsz,
                                            batch_size // WORLD_SIZE,
                                            gs,
                                            single_cls,
                                            hyp=hyp,
                                            augment=True,
                                            cache=None,
                                            rect=False,
                                            rank=-1,
                                            workers=1,
                                            image_weights=False,
                                            quad=False,
                                            prefix=colorstr('train: '),
                                            shuffle=True)
# Create Testloader
val_path = "/home/raytjchen/Desktop/code/datasets/coco/images/val2017"
val_loader = create_dataloader(
                            val_path,
                            imgsz,
                            batch_size // WORLD_SIZE * 2,
                            gs,
                            single_cls,
                            hyp=hyp,
                            cache=None,
                            rect=True,
                            rank=-1,
                            workers=1,
                            pad=0.5,
                            prefix=colorstr('val: '))[0]
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
max_epochs = 1
evaluator = Lightning(
    DetectionModule(epochs=max_epochs, hyp=hyp, batch_size=batch_size),
    Trainer(
        gpus=1,
        max_epochs=max_epochs,
        fast_dev_run=True,
        #accumulate_grad_batches = 4,
    ),
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
from nni.retiarii.strategy import DARTS as DartsStrategy
strategy = DartsStrategy()
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
config = RetiariiExeConfig(execution_engine='oneshot')
cfg="./models/yolov5s_nas.yaml"
model_space = Model(cfg=cfg, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)
exported_arch = experiment.export_top_models()[0]
print(exported_arch)
with fixed_arch(exported_arch):
    final_model = Model(cfg=cfg, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
'''