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
    def __init__(self, root:str , mode="train", transforms=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transforms = transforms

        self.files_directory = self.root

        self.filenames = self._read_split()  # read train/valid/test splits
        
        #changing self classes also update classes below in pre process
        self.classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic
        self.blender_names = {0:'background',  40:'small allu', 88:'plastic tube', 112:'large allu', 136:'v plastic', 184:'large nut', 208:'bolt', 232:'small nut'}
        #self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt'}
        #self.label_names = ['background',  'small allu', 'plastic tube', 'large allu', 'large nut', 'bolt']
        self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt', 6:'small nut'}
        self.label_names = ['background',  'small allu', 'plastic tube', 'large allu', 'large nut', 'bolt', 'small nut']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        file_path = os.path.join(self.files_directory, filename)
        
        with h5py.File(file_path, 'r') as data: 
            image = np.array(data['colors'])
            mask = np.array(data['class_segmaps'])
       
        mask = self._preprocess_mask(mask)

        sample = {}
        if self.transforms is not None:
            transformed = self.transforms(image=image, 
                                          mask=mask,
                                          depth=None)

            sample['image'] = transformed['image']
            sample['mask'] = transformed['mask'].long()

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:6} # naming both plastic tube and plastic

        mask = mask.astype(np.float32)
        #Remove this and do it in before loading data and save as h5p5
        for c in classes:
            mask[mask==c] = classes[c]

        return mask

    def _read_split(self):
        
        filenames = [f for f in os.listdir(self.files_directory) if os.path.isfile(os.path.join(self.files_directory, f))]
        #if self.mode == "train":  # 90% for train
        #    filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        #elif self.mode == "valid":  # 10% for validation
        #    filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]

        print ("Loaded image for "+self.mode+" : " , len(filenames))
        return filenames
#======================================================================


class SequentialRobocupDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transforms=None):

        assert mode in {"train", "valid", "test", "two_sequence", "three_sequence"}

        self.root = root
        self.mode = mode
        self.transforms = transforms

        self.files_directory = self.root

        self.filenames = self._read_split()  # read train/valid/test splits
        
        #changing self classes also update classes below in pre process
        self.classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic
        self.blender_names = {0:'background',  40:'small allu', 88:'plastic tube', 112:'large allu', 136:'v plastic', 184:'large nut', 208:'bolt', 232:'small nut'}
        #self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt'}
        #self.label_names = ['background',  'small allu', 'plastic tube', 'large allu', 'large nut', 'bolt']
        self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt', 6:'small nut'}
        self.label_names = ['background',  'small allu', 'plastic tube', 'large allu', 'large nut', 'bolt', 'small nut']
        #Saving Camera Matrix K and its inverse
        if self.mode ==  "train" or self.mode == "valid": 
            filename = self.filenames[0]
        elif self.mode == "two_sequence" or self.mode == "three_sequence":
            filename = self.filenames[0][0]
        print ("filename ", filename)
        file_path = os.path.join(self.files_directory, filename)

        with h5py.File(file_path, 'r') as data:
            transformation_matrix = json.loads(np.array(data['camera_pose']).item().decode()) 
            
        self.K = np.array(transformation_matrix['cam_K']).reshape(3,3)
        self.Kinv= np.linalg.inv(self.K)
        
        self.height = 512 #ToDo replace this with self.K height value. Curently for blenderproc dataset there is a mismatch
        self.width = 512 #ToDo replace this with self.K width value. Curently for blenderproc dataset there is a mismatch

        

    def __len__(self):
        return len(self.filenames)
    
    def print_filenames(self):
        return self.filenames

    
    @staticmethod
    def _give_rotation_translation(old_transformation_matrix,
                                 new_transformation_matrix):
        R = np.array(old_transformation_matrix['cam_R_w2c']).reshape(3,3)
        T = np.array(old_transformation_matrix['cam_t_w2c']).reshape(3,1)
        old_pose = np.hstack((R,T))
        old_pose = np.vstack((old_pose,[0., 0., 0., 1.]))
        
        R = np.array(new_transformation_matrix['cam_R_w2c']).reshape(3,3)
        T = np.array(new_transformation_matrix['cam_t_w2c']).reshape(3,1)
        new_pose = np.hstack((R,T))
        new_pose = np.vstack((new_pose,[0., 0., 0., 1.]))
        
        tm = TransformManager()
        tm.add_transform("world", "old_pose", old_pose)
        tm.add_transform("world", "new_pose", new_pose)
        old2new = tm.get_transform("old_pose", "new_pose")
        R = old2new[:3,:3]
        T = old2new[:3,3].reshape(3,1)
        return R, T
    
    def __getitem__(self, idx):

        if self.mode == "train" or self.mode == "valid": 
            filename = self.filenames[idx]
            file_path = os.path.join(self.files_directory, filename)

            with h5py.File(file_path, 'r') as data: 
                image = np.array(data['colors'])
                mask = np.array(data['class_segmaps'])


            #trimap = np.array(Image.open(mask_path))
            mask = self._preprocess_mask(mask)

            sample = dict(image=image, mask=mask)
            if self.transforms is not None:
                sample = self.transforms(**sample)
        elif self.mode == "two_sequence" or self.mode == "three_sequence":
            #For sequence
            # send all the n images
            # only last frame maske
            # all the poses 
            seq_filenames = self.filenames[idx]
            
            images = []
            maskes = []
            depths = []
            transformation_matrices = []
            for filename in seq_filenames:
                file_path = os.path.join(self.files_directory, filename)

                with h5py.File(file_path, 'r') as data: 
                    images.append(np.array(data['colors']))
                    maskes.append( self._preprocess_mask(np.array(data['class_segmaps'])))
                    depths.append(np.array(data['depth']))
                    transformation_matrices.append(json.loads(np.array(data['camera_pose']).item().decode()))


            
            if self.mode == "two_sequence":
                try :
                    rotation_old_to_new_camera_frame, \
                    translation_old_to_new_camera_frame = self._give_rotation_translation(transformation_matrices[0], 
                                                                                      transformation_matrices[1])
                except: 
                    print ("Transformation error in sequence : ", seq_filenames)
                    return None
                sample = dict(image0 = image_to_tensor(images[0]) / 255.0,
                              depth0 = image_to_tensor(depths[0]).squeeze(dim=0),
                              image1 = image_to_tensor(images[1]) / 255.0, 
                              mask0 =  image_to_tensor(maskes[0]).long(), 
                              mask1 =  image_to_tensor(maskes[1]).long(), 
                              rotation_0_to_1_camera_frame = rotation_old_to_new_camera_frame,
                              translation_0_to_1_camera_frame = translation_old_to_new_camera_frame)

            elif self.mode == "three_sequence":
                try :
                    rotation_0_to_1_frame, \
                    translation_0_to_1_camera_frame = self._give_rotation_translation(transformation_matrices[0], 
                                                                                      transformation_matrices[1])
                    rotation_1_to_2_frame, \
                    translation_1_to_2_camera_frame = self._give_rotation_translation(transformation_matrices[1], 
                                                                                      transformation_matrices[2])
                except: 
                    print ("Transformation error in sequence : ", seq_filenames)
                    return None
                sample = dict(image0 = images[0],
                              depth0 = depths[0],
                              image1 = images[1],
                              depth1 = depths[1],
                              image2 = images[2],
                              mask0 = maskes[0], 
                              mask1 = maskes[1],
                              mask2 = maskes[2],
                              rotation_0_to_1_frame = rotation_0_to_1_frame,
                              translation_0_to_1_camera_frame = translation_0_to_1_camera_frame, 
                              rotation_1_to_2_frame = rotation_1_to_2_frame,
                              translation_1_to_2_camera_frame=translation_1_to_2_camera_frame)
                raise NotImplemented

        else:
            raise NotImplementedError("check mode variable while instantiating")

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:6} # naming both plastic tube and plastic

        mask = mask.astype(np.float32)
        #Remove this and do it in before loading data and save as h5p5
        for c in classes:
            mask[mask==c] = classes[c]

        return mask
    
    
    
    
    def _read_split(self):
        def _atoi(text):
            return int(text) if text.isdigit() else text

        def _natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ _atoi(c) for c in  text.split('.') ]

        
        filenames = [f for f in os.listdir(self.files_directory) if os.path.isfile(os.path.join(self.files_directory, f))]
        print ("Found {:d} files in the folder".format(len(filenames)))
        
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        elif self.mode == "three_sequence": 
            #Sorting files as per the numbering
            # Assumption that files are numbered '0.hdf5', '1.hdf5', '10.hdf5', '
            # Assumption there is 3 images in sequnce starting from 0 
            filenames.sort(key=_natural_keys)
            
            filenames = [[filenames[i+n] for n in range(3)] 
                         for i, x in enumerate(filenames) if i % 3 == 0]
        elif self.mode == "two_sequence": 
            #Sorting files as per the numbering
            # Assumption that files are numbered '0.hdf5', '1.hdf5', '10.hdf5', '
            filenames.sort(key=_natural_keys)
            
            # 3 camera poses can be divided into 2 frames each
            # for example 1 - 2- 3 images
            # can be divided in to [1,2] and [2,3] sequence
            filenames = [[filenames[i+j+n] for n in range(2)] 
                         for i, x in enumerate(filenames) if i % 3 == 0
                         for j in range(2)] 
                        
            print( "Found {:d} two image sequences ".format(len(filenames)))
        return filenames
