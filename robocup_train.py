import robocup_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch

dataset_path = '/scratch/dnair2m/images_robocup/'
#dataset_path = '/home/deebuls/Documents/phd/blender-dataset/learning_blenerproc/images_robocup'

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


model = robocup_model.RoboCupModel("FPN", "resnet18", in_channels=3, out_classes=6, dataset_path=dataset_path)
#model = robocup_model.RoboCupModel("FPN", "timm-mobilenetv3_small_minimal_100", in_channels=3, out_classes=6, dataset_path=dataset_path)
#model = robocup_model.RoboCupModel("Unet", "resnet18", in_channels=3, out_classes=6, dataset_path=dataset_path)

logger = TensorBoardLogger("lightning_logs", name="robocup")
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1,
    max_epochs=100,
    callbacks=[LearningRateMonitor(logging_interval="step"), 
               TQDMProgressBar(refresh_rate=1000)],
    check_val_every_n_epoch=50,
)

trainer.fit(
    model
) 
trainer.validate(
    model
) 
                                        
