import model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
import torch

dataset_path = '/scratch/dnair2m/virtual_kitti_h5/'
#dataset_path = '/home/deebuls/Documents/phd/dataset/virtual_kitti_h5/'

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

vkitti_model = model.VirtualKittiModel("Unet", "resnet18", in_channels=3, out_classes=7, dataset_path=dataset_path)

trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1,
    max_epochs=100,
    callbacks=[LearningRateMonitor(logging_interval="step"), 
               TQDMProgressBar(refresh_rate=1000)],
    check_val_every_n_epoch=30,
    #overfit_batches=2000
)

trainer.fit(
    vkitti_model
) 
trainer.validate(
    vkitti_model
) 
                                        
MODEL_PATH = './vkitti_cluster_trained.pt'
torch.save(vkitti_model.model, MODEL_PATH)

#sequence_1d_model = model.SequenceVkitiModel(model_path=MODEL_PATH,
#                                             dataset_path=dataset_path)                                       
#    
#trainer = pl.Trainer(
#    accelerator='gpu', 
#    devices=1,
#    max_epochs=6,
#    callbacks=[LearningRateMonitor(logging_interval="step"), 
#               TQDMProgressBar(refresh_rate=1000)],
#    check_val_every_n_epoch=5,
#)
#
##trainer.validate(
##    vkitti_model
##)
#
#
#trainer.fit(
#    sequence_1d_model
#) 
#
#trainer.validate(
#    sequence_1d_model
#) 
#
