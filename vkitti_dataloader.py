import os
import torch
import h5py
from PIL import Image
import io

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

class Preprocess(torch.nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, resize_shape=512) -> None:
        super().__init__()
        #self.resize = Resize(size=(resize_shape,resize_shape))
        #self.crop = RandomCrop(size=(64,64), cropping_mode="slice")

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, 
                image: np.array, 
                mask: np.array,
                depth: np.array) -> Tensor:
        #x_tmp: np.ndarray = np.array(image)  # HxWxC
        x_out: Tensor = image_to_tensor(image, keepdim=True)  # CxHxW
        #x_out: Tensor = self.resize(x_out.float()).squeeze(dim=0)
        #x_tmp: np.ndarray = np.array(mask)  # HxWxC
        mask_out: Tensor = image_to_tensor(mask).squeeze(dim=0)
        #mask_out: Tensor = self.resize(mask_out.float()).squeeze(dim=0).squeeze(dim=0)
        
        if depth is not None:
            depth_out: Tensor = image_to_tensor(depth).squeeze(dim=0)
            #depth_out: Tensor = self.resize(depth_out.float()).squeeze(dim=0).squeeze(dim=0)
        else:
            depth_out = None
        
        return {'image':x_out.float() / 255.0, 'mask':mask_out.long(), 'depth':depth_out}
    
    
class SequentialImageVirtualKittiDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str, 
        mode: str = "train", 
        transforms = None):

        assert mode in {"train", "valid", "test"}
        
        self.mode = mode
        self.transforms = transforms

        self.files_directory = root
  
        self.data_column_names=['scene', 'scenario', 'camera_number', 'frame_number', 'extrinsic']
        #if you want a subset of the data
        self.subset = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right','clone', 'fog']
        self.val_subset = ['morning', 'overcast', 'rain', 'sunset']
        
        #Filenames extracted as a pandas dataframe
        self.seq_filenames = self._read_split()  # read train/valid/test splits
        self.mask_colors = pd.read_csv(os.path.join(self.files_directory, 
                                               'colors.txt'), delimiter=' ')
        self.mask_colors['mask_label'] = self.mask_colors.index
        #Replacing the follwing
        '''
        [['Terrain',     0   1],
         ['Sky',         1   2],
         ['Tree',        2   3],
         ['Vegetation'   3   3],
         ['Building',    4   4],
         ['Road',        5   5],
         ['GuardRail',   6   0],
         ['TrafficSign', 7   0],
         ['TrafficLight',8   0],
         ['Pole',        9   0],
         ['Misc', ,      10  0],
         ['Truck',       11  6],
         ['Car', ,       12  6],
         ['Van',         13  6],
         ['Undefined',   14  0]]
        '''
        current_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        replace_labels = [1,2,3,3,4,5,0,0,0,0, 0, 6, 6, 6, 0]
        self.mask_colors['mask_label'] =self.mask_colors['mask_label'].replace(current_labels, 
                                                                               replace_labels)
        print ("Total classes ", len(self.mask_colors['mask_label'].unique()))
        print ("Total classes ", (self.mask_colors['mask_label'].unique()))
        #Final classes are
        self.mask_colors = self.mask_colors.values.tolist()
        self.label_names = ['misc', 'Terrain', 'Sky', 'Tree', 'Building', 'Road', 'Vehicle']
      
    def __len__(self) -> int:
        return len(self.seq_filenames)
    
    @staticmethod
    def _give_rotation_translation(old_transformation_matrix,
                                 new_transformation_matrix):
        
        tm = TransformManager()
        tm.add_transform("world", "old_pose", old_transformation_matrix)
        tm.add_transform("world", "new_pose", new_transformation_matrix)
        old2new = tm.get_transform("old_pose", "new_pose")
        R = old2new[:3,:3]
        T = old2new[:3,3].reshape(3,1)
        return R, T
    
    def __getitem__(self, index: int) -> dict:

        len_sequence = len(self.seq_filenames[index])
        transformation_matrices = None
        sample = {}
        for idx, filename in enumerate(self.seq_filenames[index]):
            sidx = str(idx)
            #Reading image as numpy
            with h5py.File(filename, 'r') as data:
                
                sample['image'+sidx] = np.copy(np.asarray(Image.open(io.BytesIO(np.array(data['image'])))))
                sample['mask'+sidx] = np.copy(np.asarray(Image.open(io.BytesIO(np.array(data['mask'])))))         
                
                #Was geting a user warning that array is not writeable and pytroch needs writeable
                #sample['image'+sidx] = np.copy(sample['image'+sidx])
                #sample['mask'+sidx] = np.copy(sample['mask'+sidx])
                if (idx+1) < len_sequence: 
                    sample['depth'+sidx] = np.copy(np.asarray(Image.open(io.BytesIO(np.array(data['depth'])))))
                    #sample['depth'+sidx] = np.copy(sample['depth'+sidx])
                    
                if transformation_matrices is None:
                    transformation_matrices=np.array(data['extrinsic'])
                else:
                    new_transformation_matrix = np.array(data['extrinsic'])
                    r,t = self._give_rotation_translation(transformation_matrices,
                                                             new_transformation_matrix)
                    sample['rotation_'+str(idx-1)+'_to_'+str(idx)+'_camera_frame'] = r
                    sample['translation_'+str(idx-1)+'_to_'+str(idx)+'_camera_frame'] = t
                    transformation_matrices = new_transformation_matrix
                


            sample['mask'+sidx] = self._preprocess_mask(sample['mask'+sidx])

            #Applies transformation and converts to tensor
            if self.transforms is not None:            
                transformed = self.transforms(image=sample['image'+sidx], 
                                              mask=sample['mask'+sidx], 
                                              depth=sample['depth'+sidx] if (idx+1) < len_sequence  else None)

                sample['image'+sidx] = transformed['image']
                sample['mask'+sidx] = transformed['mask'].long()
                if (idx+1) < len_sequence:
                    sample['depth'+sidx] = transformed['depth']
        

        return sample
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        ''' 
        Convert RGB mask to single channel mask based on the color value
        provided in color.txt file 
        
        Parameters:
            mask: Numpy array mask of shape [height, width, 3]
            out: Numpy array of shape [height, width]
        '''
        preprocessed_mask = np.zeros(mask.shape[:2])
        for index, row in enumerate(self.mask_colors):
            # The columns of  mask_color dataframe is ['Terrain', r, g, b, mask_label]
            idx = np.all(mask == (row[1], row[2], row[3]), axis=-1) #
            preprocessed_mask[idx] = row[4]

        return preprocessed_mask

    def _read_split(self) -> list:
        ''' 
        Parses the virual kitti dataset and converts to a pandas dataframe
        
        Parameters:
            out: A list
        '''

        filenames = pd.read_csv(self.files_directory+'/virtual_kiti_file_naming.csv')
        if self.mode == "train":  # 90% for train
            filenames = filenames[filenames['scenario'].isin(self.subset)]
        elif self.mode == "valid":  # 10% for validation
            filenames = filenames[filenames['scenario'].isin(self.val_subset)]
        
        scene_folder_name = filenames['scene'].unique()
        scenario_folder_name = filenames['scenario'].unique()
        camera_folder_name = filenames['camera_number'].unique()
        frame_names = filenames['frame_number'].unique()
        
        #getting sequence of 2 frames here 
        #change here for 3 or n frames 
        seq_frame_names = [[i+n for n in range(0, 2, 1)] for i, x in enumerate(frame_names)] 
        
        #Runs loop and adds filenames
        seq_filenames = []
        for scene in scene_folder_name:
            for scenario in scenario_folder_name:
                for c in camera_folder_name:
                    for seq in seq_frame_names:
                        data_0 = scene+'_'+scenario+'_'+c+'_'+str(seq[0]).zfill(5)+'.h5'
                        data_1 = scene+'_'+scenario+'_'+c+'_'+str(seq[1]).zfill(5)+'.h5'

                        data_0 = os.path.join(self.files_directory, data_0)
                        data_1 = os.path.join(self.files_directory, data_1)
                        if os.path.exists(data_0) and os.path.exists(data_1):
                            seq_filenames.append([data_0, data_1])
                            
        print( "Found {:d} 2 single images sequences".format(len(seq_filenames)))
                            
        print ("Selecting {:d} two image sequences for mode {:s}".format(len(seq_filenames), self.mode))
        return seq_filenames


class SingleImageVirtualKittiDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str, 
        mode: str = "train", 
        transforms = None):

        assert mode in {"train", "valid", "test"}
        
        self.mode = mode
        self.transforms = transforms

        self.files_directory = root
  
        self.data_column_names=['scene', 'scenario', 'camera_number', 'frame_number', 'extrinsic']
        #if you want a subset of the data
        self.subset = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right','clone', 'fog']
        self.val_subset = ['morning', 'overcast', 'rain', 'sunset']
        
        #Filenames extracted as a pandas dataframe
        self.filenames = self._read_split()  # read train/valid/test splits
        self.mask_colors = pd.read_csv(os.path.join(self.files_directory, 
                                               'colors.txt'), delimiter=' ')
        self.mask_colors['mask_label'] = self.mask_colors.index
        #Replacing the follwing
        '''
        [['Terrain',     0   1],
         ['Sky',         1   2],
         ['Tree',        2   3],
         ['Vegetation'   3   3],
         ['Building',    4   4],
         ['Road',        5   5],
         ['GuardRail',   6   0],
         ['TrafficSign', 7   0],
         ['TrafficLight',8   0],
         ['Pole',        9   0],
         ['Misc', ,      10  0],
         ['Truck',       11  6],
         ['Car', ,       12  6],
         ['Van',         13  6],
         ['Undefined',   14  0]]
        '''
        current_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        replace_labels = [1,2,3,3,4,5,0,0,0,0, 0, 6, 6, 6, 0]
        self.mask_colors['mask_label'] =self.mask_colors['mask_label'].replace(current_labels, 
                                                                               replace_labels)
        print ("Total classes ", len(self.mask_colors['mask_label'].unique()))
        print ("Total classes ", (self.mask_colors['mask_label'].unique()))
        #Final classes are
        self.mask_colors = self.mask_colors.values.tolist()
        self.label_names = ['misc', 'Terrain', 'Sky', 'Tree', 'Building', 'Road', 'Vehicle']
      
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> dict:

        _, scene, scenario, camera, frame_number, x = self.filenames[idx]
        data_filename = scene+'_'+scenario+'_'+camera+'_'+str(frame_number).zfill(5)+'.h5'
        data_filename = os.path.join(self.files_directory, data_filename)
        sample = {}
        #Reading image as numpy
        with h5py.File(data_filename, 'r') as data: 
            sample['image'] = np.asarray(Image.open(io.BytesIO(np.array(data['image']))))
            sample['mask'] = np.asarray(Image.open(io.BytesIO(np.array(data['mask']))))
            #sample['depth'] = np.asarray(Image.open(io.BytesIO(np.array(data['depth']))))
            transformation_matrices=np.array(data['extrinsic'])
            #Was geting a user warning that array is not writeable and pytroch needs writeable
            sample['image'] = np.copy(sample['image'])
            sample['mask'] = np.copy(sample['mask'])
            #sample['depth'] = np.copy(sample['depth'])
            

        sample['mask'] = self._preprocess_mask(sample['mask'])
        
        #Applies transformation and converts to tensor
        if self.transforms is not None:            
            transformed = self.transforms(image=sample['image'], 
                                          mask=sample['mask'],
                                          depth=None)

            sample['image'] = transformed['image']
            sample['mask'] = transformed['mask'].long()
        

        return sample
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        ''' 
        Convert RGB mask to single channel mask based on the color value
        provided in color.txt file 
        
        Parameters:
            mask: Numpy array mask of shape [height, width, 3]
            out: Numpy array of shape [height, width]
        '''
        preprocessed_mask = np.zeros(mask.shape[:2])
        for index, row in enumerate(self.mask_colors):
            # The columns of  mask_color dataframe is ['Terrain', r, g, b, mask_label]
            idx = np.all(mask == (row[1], row[2], row[3]), axis=-1) #
            preprocessed_mask[idx] = row[4]

        return preprocessed_mask

    def _read_split(self) -> list:
        ''' 
        Parses the virual kitti dataset and converts to a pandas dataframe
        
        Parameters:
            out: A list
        '''

        filenames = pd.read_csv(self.files_directory+'/virtual_kiti_file_naming.csv')
                
        if self.mode == "train":  # 90% for train
            # Creating a dataframe with 50%
            #filenames = filenames.sample(frac = 0.6, random_state=55)
            filenames = filenames[filenames['scenario'].isin(self.subset)]
        elif self.mode == "valid":  # 10% for validation
            #sampling the same files with the random_state and droping them
            #train_filenames = filenames.sample(frac = 0.6, random_state=55)
            #filenames = filenames.drop(train_filenames.index)
            filenames = filenames[filenames['scenario'].isin(self.val_subset)]
            
        return filenames.values.tolist()


