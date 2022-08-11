import os
import torch
import h5py
from PIL import Image
import io
import json

from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomThinPlateSpline
from kornia.augmentation import RandomVerticalFlip, RandomHorizontalFlip, Resize, RandomCrop, RandomMotionBlur
from kornia.augmentation import RandomEqualize, RandomGaussianBlur, RandomGaussianNoise, RandomSharpness
import kornia as K

from torch import Tensor
import numpy as np
import pandas as pd
from kornia.augmentation import Resize

from pytransform3d.transform_manager import TransformManager

class RoboCupDataset(torch.utils.data.Dataset):
    def __init__(self, root:str , mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.files_directory = self.root

        self.filenames = self._read_split()  # read train/valid/test splits
        
        #changing self classes also update classes below in pre process
        self.classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic
        self.blender_names = {0:'background',  40:'small allu', 88:'plastic tube', 112:'large allu', 136:'v plastic', 184:'large nut', 208:'bolt', 232:'small nut'}
        self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt'}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        file_path = os.path.join(self.files_directory, filename)
        
        with h5py.File(file_path, 'r') as data: 
            image = np.array(data['colors'])
            mask = np.array(data['class_segmaps'])
            
        
        mask = self._preprocess_mask(mask)

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic

        mask = mask.astype(np.float32)
        #Remove this and do it in before loading data and save as h5p5
        for c in classes:
            mask[mask==c] = classes[c]

        return mask

    def _read_split(self):
        
        filenames = [f for f in os.listdir(self.files_directory) if os.path.isfile(os.path.join(self.files_directory, f))]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames


