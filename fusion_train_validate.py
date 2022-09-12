import vkitti_model
import robocup_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor,LearningRateMonitor,TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import argparse
import os

print(os.environ["SLURM_JOB_ID"])
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]

#GLOBAL CONSTANTS
#DATASET_PATH = '/home/deebuls/Documents/phd/dataset/virtual_kitti_h5/'
BASE_DIR = '/scratch/dnair2m/multi-view-trained-models'
MODEL_DIR = os.path.join(BASE_DIR, SLURM_JOB_ID)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    choices=['vkitti', 'robocup'],
                    help='select dataset vkitti or robocup')
    ap.add_argument('--architecture',
                    choices=['resnet18', 'resnest', 'efficientnet', 'mobilenet'],
                    help='select architecture resnet18 or regnext or mobilenet')
    ap.add_argument('--slurm_id',
                    default=None,
                    help='Provide folder name (SLURM ID) to test')
    #ADD MAX EPOCHS ARGUMENT
    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    OLD_SLURM_JOB_ID = args.slurm_id
    MODEL_DIR = os.path.join(BASE_DIR, OLD_SLURM_JOB_ID)
    print ("OLD Directory to load : ", MODEL_DIR)
    assert os.path.isdir(MODEL_DIR)
        

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    if args.architecture == 'resnet18':
        ENCODER_NAME = 'resnet18' 
    elif args.architecture == 'resnest':
        ENCODER_NAME = 'timm-resnest14d' 
    elif args.architecture == 'efficientnet':
        ENCODER_NAME = 'efficientnet-b1'
    elif args.architecture == 'mobilenet':
        ENCODER_NAME = 'timm-mobilenetv3_small_minimal_100'
    else:
        raise Exception("Wrong --architecture argument")

    if args.dataset == 'vkitti':
        ARCH_NAME = 'Unet' 
        OUT_CLASSES = 7
        TRAIN_DATASET_PATH = '/scratch/dnair2m/virtual_kitti_h5/'
        model = vkitti_model.VirtualKittiModel(ARCH_NAME,
                                        ENCODER_NAME, 
                                        in_channels=3, 
                                        out_classes=OUT_CLASSES, 
                                        dataset_path=TRAIN_DATASET_PATH)
    elif args.dataset == 'robocup':
        ARCH_NAME = 'FPN' 
        #OUT_CLASSES = 6
        OUT_CLASSES = 7 #added nut class for incresing difficulty
        TRAIN_DATASET_PATH = '/scratch/dnair2m/images_robocup/'
        VAL_DATASET_PATH = '/scratch/dnair2m/images_pose_robocup/'
        model = robocup_model.RoboCupModel(ARCH_NAME,
                                            ENCODER_NAME, 
                                            in_channels=3, 
                                            out_classes=OUT_CLASSES, 
                                            train_dataset_path=TRAIN_DATASET_PATH,
                                            valid_dataset_path=VAL_DATASET_PATH)
    else:
        raise Exception("Wrong --dataset argument")

    MODEL_PATH = MODEL_DIR+'/'+args.dataset+'_'+args.architecture+'.pt'
    logger = TensorBoardLogger("lightning_logs", name=args.dataset, version=SLURM_JOB_ID, sub_dir="train")

    #################################################################################
    if args.dataset == 'vkitti':
        sequence_1d_model = vkitti_model.SequenceVkitiModel(model_path=MODEL_PATH,
                                                     dataset_path=TRAIN_DATASET_PATH,
                                                     encoder_name=ENCODER_NAME,
                                                     convolution_type='1D')                                       
    elif args.dataset == 'robocup':
        sequence_1d_model = robocup_model.SequenceRobocupModel(model_path=MODEL_PATH,
                                                     train_dataset_path=TRAIN_DATASET_PATH,
                                                     valid_dataset_path=VAL_DATASET_PATH,
                                                     encoder_name=ENCODER_NAME,
                                                     convolution_type='1D',
                                                     out_classes=OUT_CLASSES)                                       
         
    else:
        raise Exception("Wrong --dataset argument")
           
    CONV_MODEL_PATH = MODEL_DIR+'/'+args.dataset+'_'+args.architecture+'_conv1d'+'.pt'
    logger = TensorBoardLogger("lightning_logs", name=args.dataset, version=SLURM_JOB_ID, sub_dir="1D")
    trainer = pl.Trainer(
        accelerator='gpu', 
        #devices=1,
        max_epochs=20,
        callbacks=[LearningRateMonitor(logging_interval="step"), 
                   TQDMProgressBar(refresh_rate=1000)],
        check_val_every_n_epoch=5,
        logger=logger,
        enable_checkpointing=False,
        #overfit_batches=2000
    )

    trainer.fit( sequence_1d_model) 
    print ("saving 1D fusion to ", CONV_MODEL_PATH)
    torch.save(sequence_1d_model.conv_1d, CONV_MODEL_PATH)
    trainer.validate( sequence_1d_model ) 
    trainer.test(sequence_1d_model)
    del sequence_1d_model

    #################################################################################
    if args.dataset == 'vkitti':
        sequence_2d_model = vkitti_model.SequenceVkitiModel(model_path=MODEL_PATH,
                                                     dataset_path=TRAIN_DATASET_PATH,
                                                     encoder_name=ENCODER_NAME,
                                                     convolution_type='2D')                                       
    elif args.dataset == 'robocup':
        sequence_2d_model = robocup_model.SequenceRobocupModel(model_path=MODEL_PATH,
                                                     train_dataset_path=TRAIN_DATASET_PATH,
                                                     valid_dataset_path=VAL_DATASET_PATH,
                                                     encoder_name=ENCODER_NAME,
                                                     convolution_type='2D',
                                                     out_classes=OUT_CLASSES)                                       
         
    else:
        raise Exception("Wrong --dataset argument")
           
    CONV_MODEL_PATH = MODEL_DIR+'/'+args.dataset+'_'+args.architecture+'_conv2d'+'.pt'
    logger = TensorBoardLogger("lightning_logs", name=args.dataset, version=SLURM_JOB_ID, sub_dir="2D")
    trainer = pl.Trainer(
        accelerator='gpu', 
        #devices=1,
        max_epochs=20,
        callbacks=[LearningRateMonitor(logging_interval="step"), 
                   TQDMProgressBar(refresh_rate=1000)],
        check_val_every_n_epoch=5,
        logger=logger,
        enable_checkpointing=False,
        #overfit_batches=2000
    )
    trainer.fit( sequence_2d_model) 
    print ("saving 2D fusion to ", CONV_MODEL_PATH)
    torch.save(sequence_2d_model.conv_1d, CONV_MODEL_PATH)
    trainer.validate( sequence_2d_model ) 
    trainer.test(sequence_2d_model)

    del sequence_2d_model


    #################################################################################
    if args.dataset == 'vkitti':
        dirichlet_model = vkitti_model.SequenceVkitiModel(model_path=MODEL_PATH,
                                                     dataset_path=TRAIN_DATASET_PATH,
                                                     encoder_name=ENCODER_NAME,
                                                     convolution_type='DIRICHLET')                                       
    elif args.dataset == 'robocup':
        dirichlet_model = robocup_model.SequenceRobocupModel(model_path=MODEL_PATH,
                                                     train_dataset_path=TRAIN_DATASET_PATH,
                                                     valid_dataset_path=VAL_DATASET_PATH,
                                                     encoder_name=ENCODER_NAME,
                                                     convolution_type='DIRICHLET',
                                                     out_classes=OUT_CLASSES)                                       
         
    else:
        raise Exception("Wrong --dataset argument")
           
    logger = TensorBoardLogger("lightning_logs", name=args.dataset, version=SLURM_JOB_ID, sub_dir="DIRICHLET")
    trainer = pl.Trainer(
        accelerator='gpu', 
        #devices=1,
        max_epochs=20,
        callbacks=[LearningRateMonitor(logging_interval="step"), 
                   TQDMProgressBar(refresh_rate=1000)],
        check_val_every_n_epoch=5,
        logger=logger,
        enable_checkpointing=False,
        #overfit_batches=2000
    )

    trainer.fit( dirichlet_model) 
    trainer.validate(dirichlet_model) 
    trainer.test(dirichlet_model)
    del dirichlet_model

