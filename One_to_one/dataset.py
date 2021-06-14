import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
import random
import torchvision
from PIL import Image
import os
import kornia


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, patch_size, aug=False):
        super(DatasetFromHdf5, self).__init__()
        self.path_size = patch_size
        self.toPIL = torchvision.transforms.ToPILImage()
        self.color_auger = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3)
        self.aug = aug
        self.files = []
        self.files_normal=[]
        for roots,dirs,files in file_path:
            for name in files:
                self.files.append(os.path.join(roots,name))
                self.files_normal.append(os.path.join('../../h5_normals',name))
        
        self.nsamples_file = len(self.files)
        self.norm=torchvision.transforms.Normalize(mean=[.406,.456,.485],std=[.225,.224,.229])

    def __getitem__(self, index):
        file_index = index // self.nsamples_file
        sample_index = index - file_index * self.nsamples_file
        h5 = h5py.File(self.files[file_index], 'r')
        # h5=h5py.File('../h5_track1/25612334_training_3.h5')
        input_image = h5['INPUT'][sample_index,:,:,:]
        depth_image=h5['DEPTH'][sample_index,:,:,:]
        #KIR too kunet koskhol,ridi
        target_image = h5["TARGET"][sample_index, :,:, :]
        shading_image=h5["SHADING"][sample_index,:,:,:]
        albedo_image=h5["ALBEDO"][sample_index,:,:,:]
        h5_normal=h5py.File(self.files_normal[file_index],'r')
        normal_image=h5_normal['NORMAL'][sample_index,:,:,:]
 
        input_image=torch.from_numpy(input_image).permute([2,0,1])
        depth_image = np.rollaxis(depth_image, 2)
        target_image = np.rollaxis(target_image, 2)
        shading_image = np.rollaxis(shading_image, 2)
        albedo_image = np.rollaxis(albedo_image, 2)
        normal_image = np.rollaxis(normal_image, 2)
        if self.aug:
            input_image=self.norm(input_image)

   
        return kornia.color.bgr_to_rgb(input_image.float()), torch.from_numpy(depth_image), kornia.color.bgr_to_rgb(torch.from_numpy(target_image).float()), kornia.color.bgr_to_rgb(torch.from_numpy(shading_image).float()),kornia.color.bgr_to_rgb(torch.from_numpy(albedo_image).float()),torch.from_numpy(normal_image).float()
        
    def __len__(self):
        return self.nsamples_file*len(self.files)




