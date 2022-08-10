import model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
import copy
import torch

dataset_path = '/scratch/dnair2m/virtual_kitti_h5/'
#dataset_path = '/home/deebuls/Documents/phd/dataset/virtual_kitti_h5/'

PRETRAINED_PATH = 'lightning_logs/version_481788/checkpoints/epoch=59-step=47880.ckpt'
vkitti_model = model.VirtualKittiModel.load_from_checkpoint(PRETRAINED_PATH,
                                                            dataset_path=dataset_path,
                                                            pretrained_ckpt_path=PRETRAINED_PATH)

MODEL_PATH = './vkitti_cluster_trained.pt'
torch.save(vkitti_model.model, MODEL_PATH)
sequence_1d_model = model.SequenceVkitiModel(model_path=MODEL_PATH,
                                             dataset_path=dataset_path)                                       
    
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1,
    max_epochs=6,
    callbacks=[LearningRateMonitor(logging_interval="step"), 
               TQDMProgressBar(refresh_rate=1000)],
    check_val_every_n_epoch=5,
    #overfit_batches=2000
    #resume_from_checkpoint="/home/dnair2m/multi-view-fusion-initial/lightning_logs/version_481766/checkpoints/epoch=19-step=15960.ckpt"
)

#trainer.validate(
#    vkitti_model
#)

#trainer.validate(
#    sequence_1d_model
#) 

trainer.fit(
    sequence_1d_model
) 

trainer.validate(
    sequence_1d_model
) 
                      
