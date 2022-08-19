import vkitti_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
import copy
import torch
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--fusion_type', choices=['1D','2D'], help='select 1D/2D')
args = ap.parse_args()



dataset_path = '/scratch/dnair2m/virtual_kitti_h5/'
#dataset_path = '/home/deebuls/Documents/phd/dataset/virtual_kitti_h5/'

#PRETRAINED_PATH = 'lightning_logs/version_481788/checkpoints/epoch=59-step=47880.ckpt'
#vk_model = vkitti_model.VirtualKittiModel.load_from_checkpoint(PRETRAINED_PATH,
#                                                            dataset_path=dataset_path,
#                                                            pretrained_ckpt_path=PRETRAINED_PATH)

#MODEL_PATH = './vkitti_cluster_trained.pt'
MODEL_PATH ='./vkitti_timm-resnest14d.pt'

if args.fusion_type == '1D':
    
    sequence_1d_model = vkitti_model.SequenceVkitiModel(model_path=MODEL_PATH,
                                                 dataset_path=dataset_path,
                                                 convolution_type='1D')                                       
        
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=10,
        callbacks=[LearningRateMonitor(logging_interval="step"), 
                   TQDMProgressBar(refresh_rate=1000)],
        check_val_every_n_epoch=5,
        #overfit_batches=2000
    )
    
    print("#################")
    print ("FITTING 1D model")
    print("#################")
    trainer.fit(
        sequence_1d_model
    ) 
    
    print("#################")
    print ("VALIDATE 1D model")
    print("#################")
    trainer.validate(
        sequence_1d_model
    ) 
                          
    torch.save(sequence_1d_model.conv_1d, '1Dconvolution.pt')

elif args.fusion_type == '2D':
    sequence_1d_model = vkitti_model.SequenceVkitiModel(model_path=MODEL_PATH,
                                                 dataset_path=dataset_path,
                                                 convolution_type='2D')                                       
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=10,
        callbacks=[LearningRateMonitor(logging_interval="step"), 
                   TQDMProgressBar(refresh_rate=1000)],
        check_val_every_n_epoch=5,
        #overfit_batches=2000
    )
    print("#################")
    print ("FITTING 2D model")
    print("#################")
    trainer.fit(
        sequence_1d_model
    ) 
    
    print("#################")
    print ("VALIDATING 2D model")
    print("#################")
    trainer.validate(
        sequence_1d_model
    ) 
                          
    torch.save(sequence_1d_model.conv_1d, '2Dconvolution.pt')
else:
    raise Exception("Select proper type 1D/2D")
