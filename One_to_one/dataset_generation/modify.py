from __future__ import print_function
import cv2
import numpy as np
import glob, os
#import matplotlib.pyplot as plt
import sys
#import time
import h5py
import random
#from scipy import ndimage
import ntpath
import matplotlib.pyplot as plt


PATCH_PATH = '/cvdata/amir/NTIRE/21/track1/h5_1/'
save_path='/cvdata/amir/NTIRE/21/track1/h5_normals/'
SIZE_INPUT = 256
SIZE_TARGET = 256
STRIDE = 128
count = 0
i = 1
total = 9334
for kir in range(4):
    with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_' +str(kir+1)+ '.h5'), 'r') as data:
      depths=data['DEPTH'][9334:,:,:,:]
    NORMALS=np.empty(np.shape(depths))
    count=0
    for depth in depths:
        # _,file_name=ntpath.split(name)
        # depth = np.load(name, allow_pickle=True)
        # depth=depth.item().get('normalized_depth')
        # depth=np.asarray(depth*255,dtype=float)
        # h,w=np.shape(depth)
        depth=np.squeeze(depth)
        h=256
        w=256
        normal=np.empty((h,w,3),dtype=float)
        d=np.empty((1,1,3))
        for i in range(h):
            for j in range(w):
                if j==1:
                    dydz=(depth[i,j+1]-depth[i,j])/2.0
                elif j==w-1:
                    dydz = (depth[i, j] - depth[i, j - 1]) / 2.0
                else:
                    dydz = (depth[i, j + 1] - depth[i, j - 1]) / 2.0
                if i==1:
                    dxdz=(depth[i+1,j]-depth[i,j])/2.0
                elif i==h-1:
                    dxdz = (depth[i, j] - depth[i-1, j]) / 2.0
                else:
                    dxdz = (depth[i+1, j] - depth[i-1, j]) / 2.0

                d=np.concatenate((np.expand_dims(-dxdz,(0,1,2)),np.expand_dims(-dydz,(0,1,2)),np.expand_dims(1.0,[0,1,2])),2)
                d=d/np.linalg.norm(d)
                normal[i,j,:]=d
        NORMALS[count,:,:,:]=(normal[:,:,1:2]*-1+1)/2
        count+=1
    with h5py.File(str(save_path + str(SIZE_INPUT) + str(9334) + '_' + 'training_albedo_' +str(kir+1)+ '.h5'), 'a') as data:
        A=data['NORMAL'][0:9334,:,:,:]
        del data['NORMAL']
        print('NORMAL deleted')
        NORMALS=np.concatenate((A,NORMALS))
        dset_NORMALS = data.create_dataset(name='NORMAL', shape=NORMALS.shape, data=NORMALS, dtype=np.float32)
        print('NORMAL created')
# h5w=h5py.File(str(save_path + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_4' + '.h5'), 'w')
# dset_NORMALS=h5w.create_dataset(name='NORMAL',shape=NORMALS.shape,data=NORMALS,dtype=np.float32)
# h5w.close()

# with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_2' + '.h5'), 'r') as data:
#   depths=data['DEPTH'][:]
# NORMALS=np.empty(np.shape(depths))
# count=0
# for depth in depths:
#     # _,file_name=ntpath.split(name)
#     # depth = np.load(name, allow_pickle=True)
#     # depth=depth.item().get('normalized_depth')
#     # depth=np.asarray(depth*255,dtype=float)
#     # h,w=np.shape(depth)
#     depth=np.squeeze(depth)
#     normal=np.empty((h,w,3),dtype=float)
#     d=np.empty((1,1,3))
#     for i in range(h):
#         for j in range(w):
#             if j==1:
#                 dydz=(depth[i,j+1]-depth[i,j])/2.0
#             elif j==w-1:
#                 dydz = (depth[i, j] - depth[i, j - 1]) / 2.0
#             else:
#                 dydz = (depth[i, j + 1] - depth[i, j - 1]) / 2.0
#             if i==1:
#                 dxdz=(depth[i+1,j]-depth[i,j])/2.0
#             elif i==h-1:
#                 dxdz = (depth[i, j] - depth[i-1, j]) / 2.0
#             else:
#                 dxdz = (depth[i+1, j] - depth[i-1, j]) / 2.0
#
#             d=np.concatenate((np.expand_dims(-dxdz,(0,1,2)),np.expand_dims(-dydz,(0,1,2)),np.expand_dims(1.0,[0,1,2])),2)
#             d=d/np.linalg.norm(d)
#             normal[i,j,:]=d
#     NORMALS[count,:,:,:]=(normal[:,:,1:2]*-1+1)/2
#     count+=1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_2' + '.h5'), 'a')
# dset_NORMALS=h5w.create_dataset(name='NORMAL',shape=NORMALS.shape,data=NORMALS,dtype=np.float32)
# h5w.close()
#
# with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_3' + '.h5'), 'r') as data:
#   depths=data['DEPTH'][:]
# NORMALS=np.empty(np.shape(depths))
# count=0
# for depth in depths:
#     # _,file_name=ntpath.split(name)
#     # depth = np.load(name, allow_pickle=True)
#     # depth=depth.item().get('normalized_depth')
#     # depth=np.asarray(depth*255,dtype=float)
#     # h,w=np.shape(depth)
#     depth = np.squeeze(depth)
#     normal=np.empty((h,w,3),dtype=float)
#     d=np.empty((1,1,3))
#     for i in range(h):
#         for j in range(w):
#             if j==1:
#                 dydz=(depth[i,j+1]-depth[i,j])/2.0
#             elif j==w-1:
#                 dydz = (depth[i, j] - depth[i, j - 1]) / 2.0
#             else:
#                 dydz = (depth[i, j + 1] - depth[i, j - 1]) / 2.0
#             if i==1:
#                 dxdz=(depth[i+1,j]-depth[i,j])/2.0
#             elif i==h-1:
#                 dxdz = (depth[i, j] - depth[i-1, j]) / 2.0
#             else:
#                 dxdz = (depth[i+1, j] - depth[i-1, j]) / 2.0
#
#             d=np.concatenate((np.expand_dims(-dxdz,(0,1,2)),np.expand_dims(-dydz,(0,1,2)),np.expand_dims(1.0,[0,1,2])),2)
#             d=d/np.linalg.norm(d)
#             normal[i,j,:]=d
#     NORMALS[count,:,:,:]=(normal[:,:,1:2]*-1+1)/2
#     count+=1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_3' + '.h5'), 'a')
# dset_NORMALS=h5w.create_dataset(name='NORMAL',shape=NORMALS.shape,data=NORMALS,dtype=np.float32)
# h5w.close()
#
# with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_4' + '.h5'), 'r') as data:
#   depths=data['DEPTH'][:]
# NORMALS=np.empty(np.shape(depths))
# count=0
# for depth in depths:
#     # _,file_name=ntpath.split(name)
#     # depth = np.load(name, allow_pickle=True)
#     # depth=depth.item().get('normalized_depth')
#     # depth=np.asarray(depth*255,dtype=float)
#     # h,w=np.shape(depth)
#     depth = np.squeeze(depth)
#     normal=np.empty((h,w,3),dtype=float)
#     d=np.empty((1,1,3))
#     for i in range(h):
#         for j in range(w):
#             if j==1:
#                 dydz=(depth[i,j+1]-depth[i,j])/2.0
#             elif j==w-1:
#                 dydz = (depth[i, j] - depth[i, j - 1]) / 2.0
#             else:
#                 dydz = (depth[i, j + 1] - depth[i, j - 1]) / 2.0
#             if i==1:
#                 dxdz=(depth[i+1,j]-depth[i,j])/2.0
#             elif i==h-1:
#                 dxdz = (depth[i, j] - depth[i-1, j]) / 2.0
#             else:
#                 dxdz = (depth[i+1, j] - depth[i-1, j]) / 2.0
#
#             d=np.concatenate((np.expand_dims(-dxdz,(0,1,2)),np.expand_dims(-dydz,(0,1,2)),np.expand_dims(1.0,[0,1,2])),2)
#             d=d/np.linalg.norm(d)
#             normal[i,j,:]=d
#     NORMALS[count,:,:,:]=(normal[:,:,1:2]*-1+1)/2
#     count+=1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_4' + '.h5'), 'a')
# dset_NORMALS=h5w.create_dataset(name='NORMAL',shape=NORMALS.shape,data=NORMALS,dtype=np.float32)
# h5w.close()


# depths=np.transpose(depths,[0,3,1,2])
# SEGMENTS=np.empty([total,256,256,1])
# for intervals in range(10):
#     SEGMENTS[np.logical_and(intervals*.1<=depths,depths<.1*(intervals+1))]=(intervals+.5)*.1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_1' + '.h5'), 'a')
# dset_SEGMENT=h5w.create_dataset(name='SEGMENT',shape=SEGMENTS.shape,data=SEGMENTS,dtype=np.float32)
# h5w.close()
#
# with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_2' + '.h5'), 'r') as data:
#   depths=data['DEPTH'][:]
# # depths=np.transpose(depths,[0,3,1,2])
# SEGMENTS=np.empty([total,256,256,1])
# for intervals in range(10):
#     SEGMENTS[np.logical_and(intervals*.1<=depths,depths<.1*(intervals+1))]=(intervals+.5)*.1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_2' + '.h5'), 'a')
# dset_SEGMENT=h5w.create_dataset(name='SEGMENT',shape=SEGMENTS.shape,data=SEGMENTS,dtype=np.float32)
# h5w.close()
#
# with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_3' + '.h5'), 'r') as data:
#   depths=data['DEPTH'][:]
# # depths=np.transpose(depths,[0,3,1,2])
# SEGMENTS=np.empty([total,256,256,1])
# for intervals in range(10):
#     SEGMENTS[np.logical_and(intervals*.1<=depths,depths<.1*(intervals+1))]=(intervals+.5)*.1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_3' + '.h5'), 'a')
# dset_SEGMENT=h5w.create_dataset(name='SEGMENT',shape=SEGMENTS.shape,data=SEGMENTS,dtype=np.float32)
# h5w.close()
#
# with h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_4' + '.h5'), 'r') as data:
#   depths=data['DEPTH'][:]
# # depths=np.transpose(depths,[0,3,1,2])
# SEGMENTS=np.empty([total,256,256,1])
# for intervals in range(10):
#     SEGMENTS[np.logical_and(intervals*.1<=depths,depths<.1*(intervals+1))]=(intervals+.5)*.1
# h5w=h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_4' + '.h5'), 'a')
# dset_SEGMENT=h5w.create_dataset(name='SEGMENT',shape=SEGMENTS.shape,data=SEGMENTS,dtype=np.float32)
# h5w.close()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


