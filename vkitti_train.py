import vkiti_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
import torch
import argparse
import os

print(os.environ["SLURM_JOB_ID"])
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]

#GLOBAL CONSTANTS
#DATASET_PATH = '/home/deebuls/Documents/phd/dataset/virtual_kitti_h5/'
MODEL_PATH = './vkitti_timm-resnest14d.pt'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    choices=['vkitti', 'robocup'],
                    help='select dataset vkitti or robocup')
    ap.add_argument('--architecture',
                    choices=['resnet18', 'resnest'],
                    help='select architecture resnet18 or regnext or mobilenet')
    ap.add_argument('--action',
                    choices=['train','validate'],
                    help='select action train/validate')
    ap.add_argument('--train_fusion',
                    choices=['False','1D','2D'],
                    default='False',
                    help='if training fusion select the fusion method 1D/2D')
    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    if args.architecture == 'resnet18':
        ENCODER_NAME = 'resnet18' 
    elif args.architecture == 'resnest':
        ENCODER_NAME = 'timm-resnest14d' 
    else:
        raise Exception("Wrong --architecture argument")

    if args.dataset == 'vitti':
        ARCH_NAME = 'Unet' 
        OUT_CLASSES = 7
        TRAIN_DATASET_PATH = '/scratch/dnair2m/virtual_kitti_h5/'
        model = vkitti_model.VirtualKittiModel(ARCH_NAME,
                                        ENCODER_NAME, 
                                        in_channels=3, 
                                        out_classes=OUT_CLASSES, 
                                        dataset_path=TRAIN_DATASET_PATH)
        sequence_1d_model = vkitti_model.SequenceVkitiModel(model_path=MODEL_PATH,
                                                     dataset_path=dataset_path,
                                                     convolution_type='1D')                                       
    elif args.dataset == 'robocup':
        ARCH_NAME = 'FPN' 
        OUT_CLASSES = 6
        TRAIN_DATASET_PATH = '/scratch/dnair2m/virtual_kitti_h5/'
        model = robocup_model.RoboCupModel(ARCH_NAME,
                                            ENCODER_NAME, 
                                            in_channels=3, 
                                            out_classes=OUT_CLASSES, 
                                            dataset_path=TRAIN_DATASET_PATH)
         
    else:
        raise Exception("Wrong --dataset argument")

    if args.train_fusion == 'False':
        logger = TensorBoardLogger("lightning_logs", name=arg.dataset, sub_dir="train")
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1,
            max_epochs=100,
            callbacks=[LearningRateMonitor(logging_interval="step"), 
                       TQDMProgressBar(refresh_rate=1000)],
            check_val_every_n_epoch=30,
            logger=logger
            #overfit_batches=2000
        )
    elif args.train_fusion == '1D':
           
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1,
            max_epochs=10,
            callbacks=[LearningRateMonitor(logging_interval="step"), 
                       TQDMProgressBar(refresh_rate=1000)],
            check_val_every_n_epoch=5,
            #overfit_batches=2000
        )



trainer.fit(
    vkitti_model
) 
trainer.validate(
    vkitti_model
) 
                                        
torch.save(vkitti_model.model, MODEL_PATH)

#sequence_1d_model = model.SequenceVkitiModel(model_path=MODEL_PATH,
#                                             dataset_path=DATASET_PATH)                                       
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
