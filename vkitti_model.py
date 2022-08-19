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

import vkitti_dataloader
import epipolar_geometry
import evidence_loss
import uncertain_fusion
import plot_prediction

from metrics import IoU, SegmentationMetric
from vkitti_dataloader import SingleImageVirtualKittiDataset 
from vkitti_dataloader import SequentialImageVirtualKittiDataset

from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomThinPlateSpline
from kornia.augmentation import RandomVerticalFlip, RandomHorizontalFlip, RandomMotionBlur
from kornia.augmentation import RandomGaussianNoise, RandomSharpness, RandomCrop
from kornia.augmentation import RandomEqualize, RandomGaussianBlur

IMG_SIZE = 256
old_k = np.array([[725.0087, 0, 620.5],
                   [0, 725.0087, 187],
                   [0, 0, 1]])

K = np.array([[725.0087*(IMG_SIZE/1242), 0, IMG_SIZE/2],
                   [0, 725.0087*(IMG_SIZE/375), IMG_SIZE/2],
                   [0, 0, 1]])

Kinv= np.linalg.inv(K)

class VirtualKittiModel(pl.LightningModule):

    def __init__(self, 
        arch='Unet', 
        encoder_name='resnet18', 
        in_channels=3, 
        out_classes=7,
        dataset_path=None,
		**kwargs
	):
        super().__init__()
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name, 
            encoder_weights = "imagenet",
            in_channels=in_channels, 
            classes=out_classes, 
            #**kwargs
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        self.epipolar_propagation = epipolar_geometry.EpipolarPropagation(K, 
                                   Kinv, 
                                   IMG_SIZE, 
                                   IMG_SIZE, 
                                   fill_empty_with_ones=True)
        self.epipolar_propagation.cuda()

        self.kornia_pre_transform = vkitti_dataloader.Preprocess() #per image convert to tensor
        self.transform = torch.nn.Sequential(
                RandomHorizontalFlip(p=0.30),
                RandomChannelShuffle(p=0.10),
                RandomThinPlateSpline(p=0.10),
                RandomEqualize(p=0.2),
                RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
                RandomGaussianNoise(mean=0., std=1., p=0.2),
                RandomSharpness(0.5, p=0.2)
            )
     
        # for image segmentation dice loss could be the best first choice
        self.dice_loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.evidence_loss_fn = evidence_loss.edl_mse_loss
        self.n_classes = out_classes
        
        self.val_0_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        self.val_1_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        self.ds_fusion_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        self.sum_fusion_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        self.mean_fusion_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        
        self.train_seg_metric = SegmentationMetric(self.n_classes).cuda()

        self.val_0_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.val_1_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.ds_fusion_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.sum_fusion_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.mean_fusion_seg_metric = SegmentationMetric(self.n_classes).cuda()        
        
        self.train_cm = torchmetrics.ConfusionMatrix(num_classes=self.n_classes, normalize='true')
        #kself.valid_cm = torchmetrics.ConfusionMatrix(num_classes=self.n_classes)
        
        self.DS_combine = uncertain_fusion.DempsterSchaferCombine(self.n_classes)
        self.mean_combine = uncertain_fusion.MeanUncertainty(self.n_classes)
        self.sum_combine = uncertain_fusion.SumUncertainty(self.n_classes)        

        self.fusion_methods = [self.DS_combine, self.mean_combine, self.sum_combine]#,self.bayesian, ]
        self.fusion_names = ['DS_combine', 'mean', 'sum']#'bayes',
        self.fusion_iou = [self.ds_fusion_iou, 
                                self.mean_fusion_iou,
                                self.sum_fusion_iou,
                                #self.bayes_fusion_iou,
                                #self.dampster_fusion_accuracy
                               ]
        self.fusion_seg_metric = [ self.ds_fusion_seg_metric, 
                                #   self.bayes_fusion_seg_metric,
                                   self.mean_fusion_seg_metric ,
                                   self.sum_fusion_seg_metric,
                                 ]
        
        self.dataset_path = dataset_path
        

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std 
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):
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
        #clamping highest dirchlet value 
        logits_mask = torch.clamp(logits_mask, max=50)


        ## DICE LOSS CALCULATION
        dice_loss = self.dice_loss_fn(logits_mask, mask)

        ## EVIDENTIAL LOSS CALCULATION
        #unroll the mask to single tensor 
        # [batch_size, height, width] -> [batch_size*height*width]
        mask = torch.ravel(mask)
        # [batch_size*height*width] -> [batch_size*height*width, n_classes] 
        mask = F.one_hot(mask.to(torch.long), self.n_classes)
        # [batch_size, n_classes, height, width] -> [batch_size,n_classes, height*width]
        #logits_mask = logits_mask.view(bs, self.n_classes, -1) 
        # [batch_size,n_classes, height*width] -> [batch_size, height*width, n_classes]
        logits_mask = logits_mask.permute(0,2,3,1)
        # [batch_size, height*width, n_classes] -> [batch_size*height*width, n_classes]
        logits_mask = logits_mask.reshape(-1, self.n_classes)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.evidence_loss_fn(logits_mask, mask, self.current_epoch, self.n_classes, 5)
		   
       
        logits_mask = torch.relu(logits_mask) + 1
        pred_mask = logits_mask.argmax(dim=1, keepdim=True)
        mask = mask.argmax(dim=1, keepdim=True)
      
        #loging confusion matrix and segmentation metrics 
        self.train_cm(pred_mask, mask)
            
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
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), 
                                               mask.long(), 
                                               mode="multiclass", 
                                               num_classes=self.n_classes)
        return {
            "loss": loss, #ToDo restur combined ones based on analysis
            "dice_loss": dice_loss.item(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            # normalize image here
            #image = (batch["image"]- self.mean) / self.std 
            image = batch["image"]
            mask = batch["mask"]
            image = self.transform(image)  # => we perform GPU/Batched data augmentation
            return {'image':image , 'mask':mask}
        else:
            return batch


    def training_epoch_end(self, outputs):
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
            f"train/per_image_iou": per_image_iou,
            f"train/dataset_iou": dataset_iou,
            f"train/evidential_loss": loss,
            f"train/dice_los": dice_loss,
        }
        
        self.log_dict(metrics, prog_bar=True)

        # turn confusion matrix into a figure (Tensor cannot be logged as a scalar)
        fig, ax = plt.subplots(figsize=(20,20))
        disp = ConfusionMatrixDisplay(confusion_matrix=self.train_cm.compute().cpu().numpy(),
                                      display_labels=self.label_names)
        disp.plot(ax=ax)
        # log figure
        self.logger.experiment.add_figure('train/confmat', fig, global_step=self.global_step)
        
        #np.save(CM_FILE_NAME, self.train_cm.compute().cpu().numpy())
        self.log("FrequencyIoU/train",
             self.train_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=False)

        self.train_seg_metric.reset()    
        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):

	    # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert batch["image0"].ndim == 4
        
        bs, num_channels, height, width = batch["image0"].size()

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        assert height % 32 == 0 and width % 32 == 0

        batch["mask0"] = batch["mask0"].unsqueeze(dim=1)
        batch["mask1"] = batch["mask1"].unsqueeze(dim=1)
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert batch["mask0"].ndim == 4
        assert batch["mask1"].ndim == 4
        
    
        logits_mask0 = self.forward(batch["image0"])
        logits_mask0= F.relu(logits_mask0) + 1  #ToDO shoudl we do relu and propagate or just propagate
        
        propagate_mask0 = self.epipolar_propagation(logits_mask0, 
                                                     batch['depth0']/100,
                                                     batch['translation_0_to_1_camera_frame'],
                                                     batch['rotation_0_to_1_camera_frame'])
        
        logits_mask1 = self.forward(batch["image1"])
        logits_mask1 = F.relu(logits_mask1) + 1
        
        self.val_0_iou.update(logits_mask0.argmax( dim=1, keepdim=True),batch["mask0"])
        self.log("val_iou/0", self.val_0_iou, prog_bar=True)
        #print ("shared ", batch["mask0"].device, logits_mask0.device, self.val_0_seg_metric.confusionMatrix.device )
        self.val_0_seg_metric.addBatch(logits_mask0.argmax( dim=1, keepdim=True),batch["mask0"])
        self.val_1_iou.update(logits_mask1.argmax( dim=1, keepdim=True),batch["mask1"])
        self.log("val_iou/1", self.val_1_iou, prog_bar=True)
        self.val_1_seg_metric.addBatch(logits_mask1.argmax( dim=1, keepdim=True),batch["mask1"])
        
        for fusion, name, iou, seg_metric in zip(self.fusion_methods, 
                                                 self.fusion_names, 
                                                 self.fusion_iou,
                                                 self.fusion_seg_metric):
         
            fusion_out = fusion(propagate_mask0, logits_mask1)
            fusion_out = fusion_out.to(self.device)
            
            iou.update(fusion_out.argmax( dim=1, keepdim=True), batch["mask1"])
            seg_metric.addBatch(fusion_out.argmax( dim=1, keepdim=True), batch["mask1"])
            self.log("val_iou/"+name+"_fusion", iou, prog_bar=True)

        self.log('val_1/Max val 1', torch.max(logits_mask1), prog_bar=False)
        self.log('val_1/Min val 1', torch.min(logits_mask1), prog_bar=False)
        self.log('val_1/val 1', torch.mean(logits_mask1), prog_bar=False)
            

    def validation_epoch_end(self, outputs):
        self.log("PixelAccuracy/val_0", 
                             self.val_0_seg_metric.pixelAccuracy(), prog_bar=False)
        self.log("MeanIoU/val_0", 
             self.val_0_seg_metric.meanIntersectionOverUnion(), prog_bar=False)
        self.log("FrequencyIoU/val_0",
             self.val_0_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=False)
        self.log("PixelAccuracy/val_1", 
             self.val_1_seg_metric.pixelAccuracy(), prog_bar=False)
        self.log("MeanIoU/val_1", 
             self.val_1_seg_metric.meanIntersectionOverUnion(), prog_bar=False)
        self.log("FrequencyIoU/val_1",
             self.val_0_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=False)
        print ("Val 1 Class Pixel Accuracy :", self.val_1_seg_metric.classPixelAccuracy())
        print ("Val 1 Mean Pixel Accuracy :", self.val_1_seg_metric.meanPixelAccuracy())
        print ("Val 1 IoU Per class :", self.val_1_seg_metric.IntersectionOverUnion())
        self.val_0_seg_metric.reset()
        self.val_1_seg_metric.reset()

        for seg_metric, fusion_name in zip(self.fusion_seg_metric, self.fusion_names):
            self.log("PixelAccuracy/"+fusion_name, 
                 seg_metric.pixelAccuracy(), prog_bar=False)
            self.log("MeanIoU/"+fusion_name, 
                 seg_metric.meanIntersectionOverUnion(), prog_bar=False)
            self.log("FrequencyIoU/"+fusion_name,
                 seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=False)
            print ("Class Pixel Accuracy "+fusion_name, seg_metric.classPixelAccuracy())
            print ("Mean Pixel Accuracy "+fusion_name, seg_metric.meanPixelAccuracy())
            print ("IoU Per class "+fusion_name, seg_metric.IntersectionOverUnion())

            seg_metric.reset()

    def test_step(self, batch, batch_idx):
        if batch_idx > 0:
            return
        print ("Testing ")
        bs, num_channels, height, width = batch["image0"].size()
        with torch.no_grad():
            self.model.eval()
            logits_mask0 = self.forward(batch["image0"])
            logits_mask0 = F.relu(logits_mask0) + 1  #ToDO shoudl we do relu and propagate or just propagate
            
            propagate_mask0 = self.epipolar_propagation(logits_mask0, 
                                                         batch['depth0']/100,
                                                         batch['translation_0_to_1_camera_frame'],
                                                         batch['rotation_0_to_1_camera_frame'])
            
            logits_mask1 = self.forward(batch["image1"])
            logits_mask1 = F.relu(logits_mask1) + 1
            fused_mask = self.DS_combine(propagate_mask0, logits_mask1)
            fused_mask = F.relu(fused_mask) +1
            
            uncertainty = self.n_classes / torch.sum(fused_mask, dim=1, keepdim=True)

            fig = plot_prediction.plot_sample( torch.argmax(logits_mask0, dim=1, keepdim=True),
                                               torch.argmax(propagate_mask0, dim=1, keepdim=True), 
                                               torch.argmax(logits_mask1, dim=1, keepdim=True),
                                               torch.argmax(fused_mask, dim=1, keepdim=True),   
                                               uncertainty,
                                               batch['mask1'],
                                               batch['image1']
                                             )
            self.logger.experiment.add_figure(f'test', fig, global_step=self.global_step)
            
        # log figure
        return
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-5, amsgrad=True)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4, last_epoch=-1)
        return {'optimizer': optimizer,'lr_scheduler':scheduler}
   
    def train_dataloader(self):
        dataset = SingleImageVirtualKittiDataset(self.dataset_path, "train", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=32)
                            #persistent_workers=True, pin_memory=True)
        self.label_names = dataset.label_names
        print ('Training dataset length : ', len(dataset) )
        return loader

    def val_dataloader(self):
        dataset = SequentialImageVirtualKittiDataset(self.dataset_path, "valid", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=10)
        self.label_names = dataset.label_names
        print ('Vaidation dataset length : ', len(dataset))
        return loader
        
    def test_dataloader(self):
        dataset = SequentialImageVirtualKittiDataset(self.dataset_path, "valid", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=10)
        self.label_names = dataset.label_names
        print ('Test dataset length : ', len(dataset))
        return loader
#====================================================================


class SequenceVkitiModel(pl.LightningModule):

    def __init__(self, model_path, dataset_path, encoder_name, convolution_type):
        super().__init__()

        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = torch.load(model_path)
        #Freezing the network
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.convolution_type = convolution_type
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # preprocessing parameteres for image
        self.n_classes = 7
        self.dataset_path = dataset_path
                        
        self.loss_fn = evidence_loss.edl_mse_loss
        
        self.epipolar_propagation = epipolar_geometry.EpipolarPropagation(K, 
                                   Kinv, 
                                   IMG_SIZE, 
                                   IMG_SIZE, 
                                   fill_empty_with_ones=True)
        self.epipolar_propagation.cuda()

        self.kornia_pre_transform = vkitti_dataloader.Preprocess() #per image convert to tensor
        
        #self.conv_1d = torch.nn.Conv2d(in_channels=2*self.n_classes, out_channels=self.n_classes, kernel_size=1)
        self.val_0_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        self.val_1_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        self.OneD_fusion_iou = IoU(n_classes=self.n_classes, reduction="micro-imagewise")
        
        self.val_0_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.val_1_seg_metric = SegmentationMetric(self.n_classes).cuda()
        self.OneD_fusion_seg_metric = SegmentationMetric(self.n_classes).cuda()
       
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.n_classes, normalize='true')
       
        if convolution_type == '1D':
            self.conv_1d = torch.nn.Sequential(
                              torch.nn.Conv2d(in_channels=2*self.n_classes, 
                                           out_channels=self.n_classes, 
                                           kernel_size=1, 
                                           device=self.device),
                            )
        elif convolution_type == '2D':
            self.conv_1d = torch.nn.Sequential(
                              torch.nn.Conv2d(in_channels=2*self.n_classes, 
                                           out_channels=self.n_classes, 
                                           kernel_size=3, 
                                           device=self.device),
                             torch.nn.Upsample(size=(256,256), mode = 'nearest') #Make the image size auto
                            )
        else:
            raise 


    
    def forward(self, batch):
        #Freezing the network
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        
        # normalize image here
        normalized_image = (batch["image0"] - self.mean) / self.std 
        logits_mask0 = self.model(normalized_image)

        propagate_mask0 = self.epipolar_propagation(logits_mask0, 
                                                     batch['depth0']/100,
                                                     batch['translation_0_to_1_camera_frame'],
                                                     batch['rotation_0_to_1_camera_frame'])
        
        normalized_image = (batch["image1"] - self.mean) / self.std 
        logits_mask1 = self.model(normalized_image)
      
        fused_mask = torch.concat((propagate_mask0,logits_mask1), dim=1)
        fused_mask = self.conv_1d(fused_mask)

        logits_mask0 = F.relu(logits_mask0) + 1
        logits_mask1 = F.relu(logits_mask1) + 1
        #fused_mask = F.relu(fused_mask) + 1
        
        return logits_mask0, propagate_mask0, logits_mask1, fused_mask
    
   
        
        
    def shared_step(self, batch, stage):
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert batch["image0"].ndim == 4
        
        bs, num_channels, height, width = batch["image0"].size()

        # Check that image dimensions are divisible by 32, 
        assert height % 32 == 0 and width % 32 == 0

        batch["mask0"] = batch["mask0"].unsqueeze(dim=1)
        batch["mask1"] = batch["mask1"].unsqueeze(dim=1)
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert batch["mask0"].ndim == 4
        assert batch["mask1"].ndim == 4
        
        logits_mask0, propagate_mask0, logits_mask1, fused_mask = self.forward(batch)
        
        fused_mask = fused_mask.permute(0,2,3,1)
        # [batch_size, height*width, n_classes] -> [batch_size*height*width, n_classes]
        fused_mask = fused_mask.reshape(-1, self.n_classes)
        mask = torch.ravel(batch["mask1"])
        # [batch_size*height*width] -> [batch_size*height*width, n_classes] 
        mask = F.one_hot(mask.to(torch.long), self.n_classes)
        #loss = self.loss_fn(fused_mask, mask, self.current_epoch, self.n_classes, 5)
        loss = F.cross_entropy(fused_mask, mask.argmax(dim=1).to(torch.long))
        self.log(f"{stage}/evidential_loss", loss, prog_bar=True)
        
        fused_mask = F.relu(fused_mask) + 1
        #confusion matrix calculation 
        self.confusion_matrix(fused_mask.argmax( dim=1, keepdim=True), mask.argmax( dim=1, keepdim=True))

        #Getting back to shape
        fused_mask = fused_mask.reshape(bs,  height, width, self.n_classes)
        fused_mask = fused_mask.permute(0,3,1,2)
        
        #Logging
        self.val_0_iou.update(logits_mask0.argmax( dim=1, keepdim=True), batch["mask0"])
        self.val_1_iou.update(logits_mask1.argmax( dim=1, keepdim=True), batch["mask1"])
        self.OneD_fusion_iou.update(fused_mask.argmax( dim=1, keepdim=True), batch["mask1"])

        self.val_0_seg_metric.addBatch(logits_mask0.argmax( dim=1, keepdim=True), batch["mask0"])
        self.val_1_seg_metric.addBatch(logits_mask1.argmax( dim=1, keepdim=True), batch["mask1"])
        self.OneD_fusion_seg_metric.addBatch(fused_mask.argmax( dim=1, keepdim=True), batch["mask1"])

        
        return loss
      

    def shared_epoch_end(self, outputs, stage):
        try :
            self.log(f"iou/{stage}/0_iou", self.val_0_iou.compute(), prog_bar=False)
            self.log(f"iou/{stage}/1_iou", self.val_1_iou.compute(), prog_bar=True)
            self.log(f"iou/{stage}/OneD_fusion_iou", self.OneD_fusion_iou.compute(), prog_bar=True)
            self.log("FrequencyIoU/"+stage+"/0", 
                             self.val_0_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=False)
            self.log("FrequencyIoU/"+stage+"/1", 
                             self.val_1_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)
            print ("Val 1 Class Pixel Accuracy :", self.val_1_seg_metric.classPixelAccuracy())
            print ("Val 1 Mean Pixel Accuracy :", self.val_1_seg_metric.meanPixelAccuracy())
            print ("Val 1 IoU Per class :", self.val_1_seg_metric.IntersectionOverUnion())
            self.log("FrequencyIoU/"+stage+"/OneD_fusion", 
                             self.OneD_fusion_seg_metric.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)
            self.log("PixelAccuracy"+stage+"/OneD_fusion", 
                 self.OneD_fusion_seg_metric.pixelAccuracy(), prog_bar=False)
            self.log("MeanIoU/"+stage+"/OneD_fusion" ,
                 self.OneD_fusion_seg_metric.meanIntersectionOverUnion(), prog_bar=False)
            print ("Class Pixel Accuracy", self.OneD_fusion_seg_metric.classPixelAccuracy())
            print ("Mean Pixel Accuracy", self.OneD_fusion_seg_metric.meanPixelAccuracy())
            print ("IoU Per class", self.OneD_fusion_seg_metric.IntersectionOverUnion())
        except:
            print("Error in the iou compute or FrequencyIou")
        self.val_0_seg_metric.reset()
        self.val_1_seg_metric.reset()
        self.OneD_fusion_seg_metric.reset()
        self.val_0_iou.reset()
        self.val_1_iou.reset()
        self.OneD_fusion_iou.reset()


        # turn confusion matrix into a figure (Tensor cannot be logged as a scalar)
        fig, ax = plt.subplots(figsize=(20,20))
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix.compute().cpu().numpy(),
                                      display_labels=self.label_names)
        disp.plot(ax=ax)
        # log figure
        self.logger.experiment.add_figure(stage+'/confmat', fig, global_step=self.global_step)
        
        #np.save(CM_FILE_NAME, self.train_cm.compute().cpu().numpy())
    
        self.confusion_matrix.reset()
        return       

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx): 
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        if batch_idx > 0:
            return
        bs, num_channels, height, width = batch["image0"].size()
        with torch.no_grad():
            self.model.eval()
            logits_mask0, propagate_mask0, logits_mask1, fused_mask = self.forward(batch)
            fused_mask = F.relu(fused_mask) +1
            uncertainty = self.n_classes / torch.sum(fused_mask, dim=1, keepdim=True)

            fig = plot_prediction.plot_sample( torch.argmax(logits_mask0, dim=1, keepdim=True),
                                               torch.argmax(propagate_mask0, dim=1, keepdim=True), 
                                               torch.argmax(logits_mask1, dim=1, keepdim=True),
                                               torch.argmax(fused_mask, dim=1, keepdim=True),   
                                               uncertainty,
                                               batch['mask1'],
                                               batch['image1']
                                             )
            self.logger.experiment.add_figure(f'test_'+self.convolution_type, fig, global_step=self.global_step)
            
        # log figure
        return

    def configure_optimizers(self):
        #optimizer=torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001, weight_decay=1e-5)
        optimizer=torch.optim.AdamW( self.conv_1d.parameters(), lr=1e-3, weight_decay=1e-5)
        #scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5, last_epoch=-1)
    
        return {'optimizer': optimizer,'lr_scheduler':scheduler}

    def train_dataloader(self):
        dataset = SequentialImageVirtualKittiDataset(self.dataset_path, "train", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=10)
        self.label_names = dataset.label_names
        print ('training dataset length : ', len(dataset))
        return loader

    def val_dataloader(self):
        dataset = SequentialImageVirtualKittiDataset(self.dataset_path, "valid", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=10)
        self.label_names = dataset.label_names
        print ('Vaidation dataset length : ', len(dataset))
        return loader

    def test_dataloader(self):
        dataset = SequentialImageVirtualKittiDataset(self.dataset_path, "valid", transforms=self.kornia_pre_transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=10)
        self.label_names = dataset.label_names
        print ('Test dataset length : ', len(dataset))
        return loader
