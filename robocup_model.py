import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import ConfusionMatrixDisplay
import segmentation_models_pytorch as smp
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robocup_dataloader import RoboCupDataset
import vkitti_dataloader
import epipolar_geometry
import evidence_loss
import uncertain_fusion

from metrics import IoU, SegmentationMetric
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomThinPlateSpline
from kornia.augmentation import RandomVerticalFlip, RandomHorizontalFlip, RandomMotionBlur
from kornia.augmentation import RandomGaussianNoise, RandomSharpness, RandomCrop
from kornia.augmentation import RandomEqualize, RandomGaussianBlur

IMG_SIZE = 512


class RoboCupModel(pl.LightningModule):

    def __init__(self, 
                 arch, 
                 encoder_name, 
                 in_channels, 
                 out_classes, 
                 dataset_path=None,
                 **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))


        # for image segmentation dice loss could be the best first choice
        self.dice_loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.loss_fn = evidence_loss.edl_mse_loss
        self.n_classes = out_classes
        self.dataset_path = dataset_path
        
        self.kornia_pre_transform = vkitti_dataloader.Preprocess() #per image convert to tensor
        self.transform = torch.nn.Sequential(
                #RandomHorizontalFlip(p=0.50),
                #RandomChannelShuffle(p=0.10),
                #RandomThinPlateSpline(p=0.10),
                #RandomEqualize(p=0.2),
                #RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
                #RandomGaussianNoise(mean=0., std=1., p=0.2),
                #RandomSharpness(0.5, p=0.2)
            )
     
        self.ignore_class = 0.0 #ignore background class  fr loss function
        
        self.train_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.train_cm = torchmetrics.ConfusionMatrix(num_classes=self.n_classes, normalize='true')


    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std 
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4
        
        bs, num_channels, height, width = image.size()


        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, height, width]
        assert mask.ndim == 3

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 255.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        ## DICE LOSS CALCULATION
        dice_loss = self.dice_loss_fn(logits_mask, mask)
        
        #unroll the tensor to single tensor 
        # [batch_size, 1, height, width] -> [batch_size*height*width]
        mask = torch.ravel(mask)
        
        #Remove pixels exculding the background loss function
        idx_only_objects = mask != self.ignore_class
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True       
        mask = F.one_hot(mask.to(torch.long), self.n_classes)  # [batch_size*height*width] -> [batch_size*height*width, n_classes]
        
        # [batch_size, n_classes, height, width] -> [batch_size,n_classes, height*width]
        logits_mask = logits_mask.view(bs, self.n_classes, -1) 
        # [batch_size,n_classes, height*width] -> [batch_size, height*width, n_classes]
        logits_mask = logits_mask.permute(0,2,1)
        # [batch_size, height*width, n_classes] -> [batch_size*height*width, n_classes]
        logits_mask = logits_mask.reshape_as(mask)
        
       

        #Fluctute between all loss and only objects loss excluding bakground
        if (stage == "train"):
            if self.current_epoch % 3 == 0:
                loss = self.loss_fn(logits_mask, mask, self.current_epoch, self.n_classes, 10)
            else:
                loss = self.loss_fn(logits_mask[idx_only_objects], mask[idx_only_objects], self.current_epoch, self.n_classes, 10)
        else:
            loss = self.loss_fn(logits_mask[idx_only_objects], mask[idx_only_objects], self.current_epoch, self.n_classes, 10)

        #print ("loss ", loss)
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        #prob_mask = logits_mask.sigmoid()
        #pred_mask = (prob_mask > 0.5).float()
        prob_mask = torch.relu(logits_mask) + 1
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        
        mask = mask.argmax(dim=1, keepdim=True)
        
        #Confusion matrix calculation
        confusion_matrix = self.train_cm(pred_mask, mask)
        
        #Changing back to original dimension for metrics calculation
        pred_mask = pred_mask.reshape(bs, 1, height, width )
        mask = mask.reshape(bs, 1, height, width)
          
        self.train_seg_metric.addBatch(pred_mask.long(), mask.long())
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multiclass", 
                                               num_classes=self.n_classes)
        

        return {
            "loss": loss,
            "dice_loss": dice_loss.item(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    #def on_after_batch_transfer(self, batch, dataloader_idx):
    #    if self.trainer.training:
    #        image = batch["image"]
    #        mask = batch["mask"]
    #        image = self.transform(image)  # => we perform GPU/Batched data augmentation
    #        return {'image':image , 'mask':mask}
    #    else:
    #        return batch

    def shared_epoch_end(self, outputs, stage):
        
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # aggregate step metics
        loss = [x["loss"].item() for x in outputs]
        loss = sum(loss)/len(loss)
        dice_loss = [x["dice_loss"] for x in outputs]
        dice_loss = sum(dice_loss)/len(dice_loss)
        
        metrics = {
            f"{stage}/per_image_iou": per_image_iou,
            f"{stage}/dataset_iou": dataset_iou,
            f"{stage}/evidential_loss": loss,
            f"{stage}/dice_los": dice_loss,
        }
        
        self.log_dict(metrics, prog_bar=True)
        # turn confusion matrix into a figure (Tensor cannot be logged as a scalar)
        fig, ax = plt.subplots(figsize=(20,20))
        disp = ConfusionMatrixDisplay(confusion_matrix=self.train_cm.compute().cpu().numpy(),
                                      display_labels=self.label_names)
        disp.plot(ax=ax)
        # log figure
        self.logger.experiment.add_figure(f'{stage}/confmat', fig, global_step=self.global_step)
        
        self.log("FrequencyIoU/"+stage,
             self.train_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=False)

        self.train_seg_metric.reset()    
        self.train_cm.reset()
        

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5, amsgrad=True)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5, last_epoch=-1)
        return {'optimizer': optimizer,'lr_scheduler':scheduler}

    def train_dataloader(self):
        dataset = RoboCupDataset(self.dataset_path, "train", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=32)
                            #persistent_workers=True, pin_memory=True)
        self.label_names = dataset.label_names
        print ('Training dataset length : ', len(dataset) )
        return loader

    def val_dataloader(self):
        dataset = RoboCupDataset(self.dataset_path, "valid", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=10)
        self.label_names = dataset.label_names
        print ('Vaidation dataset length : ', len(dataset))
        return loader
        


#====================================================================

