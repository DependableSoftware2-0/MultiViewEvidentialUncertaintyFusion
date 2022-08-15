import model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
import copy
import torch

dataset_path = '/scratch/dnair2m/virtual_kitti_h5/'
#dataset_path = '/home/deebuls/Documents/phd/dataset/virtual_kitti_h5/'

#PRETRAINED_PATH = 'lightning_logs/version_481788/checkpoints/epoch=59-step=47880.ckpt'
#vkitti_model = model.VirtualKittiModel.load_from_checkpoint(PRETRAINED_PATH,
#                                                            dataset_path=dataset_path,
#                                                            pretrained_ckpt_path=PRETRAINED_PATH)

MODEL_PATH = './vkitti_cluster_trained.pt'
#torch.save(vkitti_model.model, MODEL_PATH)
#sequence_1d_model = model.SequenceVkitiModel(model_path=MODEL_PATH,
#                                             dataset_path=dataset_path,
#                                             convolution_type='1D')                                       
#    
#trainer = pl.Trainer(
#    accelerator='gpu', 
#    devices=1,
#    max_epochs=10,
#    callbacks=[LearningRateMonitor(logging_interval="step"), 
#               TQDMProgressBar(refresh_rate=1000)],
#    check_val_every_n_epoch=5,
#    #overfit_batches=2000
#)
#
##trainer.validate(
##    vkitti_model
##)
#print("#################")
#print ("FITTING 1D model")
#print("#################")
#trainer.fit(
#    sequence_1d_model
#) 
#
#print("#################")
#print ("VALIDATE 1D model")
#print("#################")
#trainer.validate(
#    sequence_1d_model
#) 
#                      
#torch.save(sequence_1d_model.conv_1d, '1Dconvolution.pt')
#
sequence_1d_model = model.SequenceVkitiModel(model_path=MODEL_PATH,
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
