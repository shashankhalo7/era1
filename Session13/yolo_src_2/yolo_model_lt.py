from __future__ import print_function

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from tqdm import tqdm
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from loss import YoloLoss



""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

model_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        

class YOLOv3(pl.LightningModule):
    def __init__(self,config,train_loader,test_loader, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

        self.config = config

        self.IMAGE_SIZE = 416
        self.S = [self.IMAGE_SIZE // 32, self.IMAGE_SIZE // 16, self.IMAGE_SIZE // 8]
        
        self.ANCHORS = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ] 
        
        self.scaled_anchors = (torch.tensor(self.ANCHORS)* torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.train_loader = train_loader
        self.test_loader = test_loader
        

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in model_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-05,weight_decay = 1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=1e-03, steps_per_epoch=len(self.train_loader), epochs=40,div_factor=100,pct_start = 5/40)
        return [optimizer],[scheduler]
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2]
        )
        
        out = self(x)
        train_loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )
        
        self.log("train_loss", train_loss,prog_bar=True, on_step=True, on_epoch=True)
        
    def on_epoch_start(self):
         self.plot_couple_examples( self.test_loader, 0.6, 0.5, self.scaled_anchors)
        
    def on_train_epoch_end(self):
        current_epoch = self.current_epoch
        if self.config.SAVE_MODEL:
            self.save_checkpoint(self.optimizers(), filename=f"checkpoint_{current_epoch}.pth.tar")
        self.check_class_accuracy(self.train_loader, threshold=self.config.CONF_THRESHOLD,logger=self.log)
        
    def on_valid_epoch_end(self):
        current_epoch = self.current_epoch
        if current_epoch % 10 == 0 or current_epoch in (1,2,3):
            self.check_class_accuracy(self, self.test_loader, threshold=self.config.CONF_THRESHOLD,logger=self.log)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_loader,
                self,
                iou_threshold=self.config.NMS_IOU_THRESH,
                anchors=self.config.ANCHORS,
                threshold=self.config.CONF_THRESHOLD,
            )
            mapval = self.mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=self.config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=self.config.NUM_CLASSES,
            )
            print(mapval)
            self.log("MAP", mapval.item(),prog_bar=True, on_step=True, on_epoch=True)
            
    def iou_width_height(self,boxes1, boxes2):
        """
        Parameters:
            boxes1 (tensor): width and height of the first bounding boxes
            boxes2 (tensor): width and height of the second bounding boxes
        Returns:
            tensor: Intersection over union of the corresponding boxes
        """
        intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
            boxes1[..., 1], boxes2[..., 1]
        )
        union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
        )
        return intersection / union


    def intersection_over_union(self,boxes_preds, boxes_labels, box_format="midpoint"):
        """
        Video explanation of this function:
        https://youtu.be/XXYG5ZWtjj0

        This function calculates intersection over union (iou) given pred boxes
        and target boxes.

        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

        Returns:
            tensor: Intersection over union for all examples
        """

        if box_format == "midpoint":
          box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
          box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
          box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
          box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
          box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
          box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
          box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
          box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        if box_format == "corners":
          box1_x1 = boxes_preds[..., 0:1]
          box1_y1 = boxes_preds[..., 1:2]
          box1_x2 = boxes_preds[..., 2:3]
          box1_y2 = boxes_preds[..., 3:4]
          box2_x1 = boxes_labels[..., 0:1]
          box2_y1 = boxes_labels[..., 1:2]
          box2_x2 = boxes_labels[..., 2:3]
          box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)


    def non_max_suppression(self,bboxes, iou_threshold, threshold, box_format="corners"):
        """
        Video explanation of this function:
        https://youtu.be/YDkjWEN8jNA

        Does Non Max Suppression given bboxes

        Parameters:
            bboxes (list): list of lists containing all bboxes with each bboxes
            specified as [class_pred, prob_score, x1, y1, x2, y2]
            iou_threshold (float): threshold where predicted bboxes is correct
            threshold (float): threshold to remove predicted bboxes (independent of IoU)
            box_format (str): "midpoint" or "corners" used to specify bboxes

        Returns:
            list: bboxes after performing NMS given a specific IoU threshold
        """

        assert type(bboxes) == list

        bboxes = [box for box in bboxes if box[1] > threshold]
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        bboxes_after_nms = []

        while bboxes:
            chosen_box = bboxes.pop(0)

            bboxes = [
                box
                for box in bboxes
                if box[0] != chosen_box[0]
                or intersection_over_union(
                    torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),
                    box_format=box_format,
                )
                < iou_threshold
            ]

            bboxes_after_nms.append(chosen_box)

        return bboxes_after_nms


    def mean_average_precision(self,
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
    ):
        """
        Video explanation of this function:
        https://youtu.be/FppOzcDvaDI

        This function calculates mean average precision (mAP)

        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes

        Returns:
            float: mAP value across all classes given a specific IoU threshold
        """

        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)


    def plot_image(self,image, boxes):
        """Plots predicted bounding boxes on the image"""
        cmap = plt.get_cmap("tab20b")
        class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
        colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
        im = np.array(image)
        height, width, _ = im.shape

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height

        # Create a Rectangle patch
        for box in boxes:
            assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
            class_pred = box[0]
            box = box[2:]
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=2,
                edgecolor=colors[int(class_pred)],
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(class_pred)], "pad": 0},
            )

        plt.show()


    def get_evaluation_bboxes(self,
        loader,
        model,
        iou_threshold,
        anchors,
        threshold,
        box_format="midpoint",
    ):
        # make sure model is in eval before get bboxes
        train_idx = 0
        all_pred_boxes = []
        all_true_boxes = []
        for batch_idx, (x, labels) in enumerate(tqdm(loader)):
            predictions = model(x)
            batch_size = x.shape[0]
            bboxes = [[] for _ in range(batch_size)]
            for i in range(3):
                S = predictions[i].shape[2]
                anchor = torch.tensor([*anchors[i]]) * S
                boxes_scale_i = cells_to_bboxes(
                    predictions[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            # we just want one bbox for each label, not one for each scale
            true_bboxes = cells_to_bboxes(
                labels[2], anchor, S=S, is_preds=False
            )

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                    box_format=box_format,
                )

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for box in true_bboxes[idx]:
                    if box[1] > threshold:
                        all_true_boxes.append([train_idx] + box)

                train_idx += 1

        return all_pred_boxes, all_true_boxes


    def cells_to_bboxes(self,predictions, anchors, S, is_preds=True):
        """
        Scales the predictions coming from the model to
        be relative to the entire image such that they for example later
        can be plotted or.
        INPUT:
        predictions: tensor of size (N, 3, S, S, num_classes+5)
        anchors: the anchors used for the predictions
        S: the number of cells the image is divided in on the width (and height)
        is_preds: whether the input is predictions or the true bounding boxes
        OUTPUT:
        converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                        object score, bounding box coordinates
        """
        BATCH_SIZE = predictions.shape[0]
        num_anchors = len(anchors)
        box_predictions = predictions[..., 1:5]
        if is_preds:
            anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
            box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
            box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
            scores = torch.sigmoid(predictions[..., 0:1])
            best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
        else:
            scores = predictions[..., 0:1]
            best_class = predictions[..., 5:6]

        cell_indices = (
            torch.arange(S)
            .repeat(predictions.shape[0], 3, S, 1)
            .unsqueeze(-1)
        )
        x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
        y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
        w_h = 1 / S * box_predictions[..., 2:4]
        converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
        return converted_bboxes.tolist()

    def check_class_accuracy(self, loader, threshold,logger):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for idx, (x, y) in enumerate(tqdm(loader)):
            x,y = x.to(self.device),y.to(self.device)
            out = self(x)

            for i in range(3):
                obj = y[i][..., 0] == 1 # in paper this is Iobj_i
                noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

                correct_class += torch.sum(
                    torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
                )
                tot_class_preds += torch.sum(obj)

                obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
                correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
                tot_obj += torch.sum(obj)
                correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
                tot_noobj += torch.sum(noobj)

        print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
        print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
        print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
        
        logger("Class accuracy",(correct_class/(tot_class_preds+1e-16))*100,prog_bar=True, on_step=False, on_epoch=True)
        logger("No obj accuracy",(correct_noobj/(tot_noobj+1e-16))*100,prog_bar=True, on_step=False, on_epoch=True)
        logger("Obj Accuracy",(correct_obj/(tot_obj+1e-16))*100,prog_bar=True, on_step=False, on_epoch=True)


    def get_mean_std(self,loader):
        # var[X] = E[X**2] - E[X]**2
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

        for data, _ in tqdm(loader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

        return mean, std


    def save_checkpoint(self, optimizer, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)


    def load_checkpoint(self,checkpoint_file, optimizer, lr):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    def get_loaders(self,train_csv_path, test_csv_path):
        from dataset_org import YOLODataset

        IMAGE_SIZE = config.IMAGE_SIZE
        train_dataset = YOLODataset(
            train_csv_path,
            transform=config.train_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        test_dataset = YOLODataset(
            test_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

        train_eval_dataset = YOLODataset(
            train_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        train_eval_loader = DataLoader(
            dataset=train_eval_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, test_loader, train_eval_loader

    def plot_couple_examples(self, loader, thresh, iou_thresh, anchors):
        x, y = next(iter(loader))
        out = self(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box


        for i in range(batch_size//4):
            nms_boxes = non_max_suppression(
                bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
            )
            plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



    def seed_everything(self,seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def clip_coords(self,boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
  
    def xywhn2xyxy(self,x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
        y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
        y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
        y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
        return y


    def xyn2xy(self,x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = w * x[..., 0] + padw  # top left x
        y[..., 1] = h * x[..., 1] + padh  # top left y
        return y

    def xyxy2xywhn(self,x, w=640, h=640, clip=False, eps=0.0):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
        if clip:
            clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
        y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
        y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
        y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
        return y

    def clip_boxes(self,boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
            
    
    def loss_fn(self, predictions, target, anchors):
        
        predictions = predictions.to(self.device)
        target = target.to(self.device)
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        device = predictions.device
        anchors = anchors.to(device)
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = self.intersection_over_union(box_preds[obj], target[..., 1:5][obj])
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )