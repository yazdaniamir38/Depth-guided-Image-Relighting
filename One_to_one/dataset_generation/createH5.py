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


Shading_PATH='../training_patches_varied_256/results/'
DATA_PATH = '../training_patches_varied_256/input/'
LABEL_PATH = '../training_patches_varied_256/target/'
PATCH_PATH = '../h5/data/'
SIZE_INPUT = 256
SIZE_TARGET = 256
STRIDE = 128
count = 0
i = 1
total = 9334
h5fw = h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_1' + '.h5'), 'w')
names=glob.glob(LABEL_PATH+'*.png')
total=len(names)//4
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
ALBEDO=np.empty(shape=(total,SIZE_INPUT,SIZE_INPUT,3))
SHADING=np.empty(shape=(total,SIZE_INPUT,SIZE_INPUT,3))
INPUT = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
TARGET = np.empty(shape=(total, SIZE_TARGET, SIZE_TARGET, 3))
DEPTH=np.empty(shape=(total,SIZE_INPUT,SIZE_INPUT,1))
k = 0
p = np.random.permutation(total)
# print(p)

# # names=names[np.random.permutation(len(names))]
# # names=name[0:39000]
#
for data_image in names[0:total]:
    string_label = path_leaf(data_image)
    string_shading='s'+string_label[3:]
    string_data= 'inp' + string_label[3:]
    string_depth='depth'+string_label[3:]
    # print(string_data)
    print(string_label)
    shading_image_name=Shading_PATH+string_shading
    data_image_name = DATA_PATH + string_data
    depth_image_name=DATA_PATH+string_depth
    #BI_img_name = BI_PATH + HR_img_name[12:19] + '.png'
    # print(label_image_name)
    imgShading=cv2.imread(shading_image_name)
    imgData = cv2.imread(data_image_name)
    imgLabel = cv2.imread(data_image)
    imgDepth=cv2.imread(depth_image_name)[:,:,0:1]
    # normalizing the input and target images
    # imgShading_normalized=cv2.cvtColor(imgShading,cv2.COLOR_BGR2RGB)/255.0
    imgShading_normalized=imgShading/255.0
    imgData_normalized = imgData/255.0
    imgLabel_normalized = imgLabel/255.0
    imgAlbedo_normalized=imgLabel_normalized/(imgShading_normalized+.0001)
    imgDepth_normalized=imgDepth/255.0

    INPUT[p[k], :, :, :] = imgData_normalized
    TARGET[p[k], :, :, :] =imgLabel_normalized
    DEPTH[p[k],:,:,:]= imgDepth_normalized
    ALBEDO[p[k],:,:,:]=imgAlbedo_normalized
    SHADING[p[k],:,:,:]=imgShading_normalized
    k = k + 1
    print(str(k) + '-INPUT' + str(INPUT.shape) + '-TARGET' + str(TARGET.shape))
    sys.stdout.flush() #?
dset_input = h5fw.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
print('>>>>INPUT file generated')
dset_depth = h5fw.create_dataset(name='DEPTH', shape=DEPTH.shape, data=DEPTH, dtype=np.float32)
dset_target = h5fw.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
dset_albedo = h5fw.create_dataset(name='ALBEDO', shape=ALBEDO.shape, data=ALBEDO, dtype=np.float32)
dset_shading = h5fw.create_dataset(name='SHADING', shape=SHADING.shape, data=SHADING, dtype=np.float32)
print('>>>>TARGET file generated')
print('>>>>save file' + 'training' + 'INPUT_' + str(SIZE_INPUT) + 'TARGET_' + str(SIZE_TARGET))
h5fw.close()



h5fw = h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_2' + '.h5'), 'r')

#
ALBEDO = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
SHADING = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
INPUT = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
TARGET = np.empty(shape=(total, SIZE_TARGET, SIZE_TARGET, 3))
DEPTH = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 1))
k = 0
p = np.random.permutation(total)
print(p)
# names = glob.glob(LABEL_PATH + '*.png')[total:total*2]
for data_image in names[total:2*total]:
    string_label = path_leaf(data_image)
    string_shading = 's' + string_label[3:]
    string_data = 'inp' + string_label[3:]
    string_depth = 'depth' + string_label[3:]
    # print(string_data)
    print(string_label)
    shading_image_name = Shading_PATH + string_shading
    data_image_name = DATA_PATH + string_data
    depth_image_name = DATA_PATH + string_depth
    # BI_img_name = BI_PATH + HR_img_name[12:19] + '.png'
    # print(label_image_name)
    imgShading = cv2.imread(shading_image_name)
    imgData = cv2.imread(data_image_name)
    imgLabel = cv2.imread(data_image)
    imgDepth = cv2.imread(depth_image_name)[:, :, 0:1]
    # normalizing the input and target images
    imgShading_normalized = cv2.cvtColor(imgShading, cv2.COLOR_BGR2RGB) / 255.0
    imgData_normalized = imgData / 255.0
    imgLabel_normalized = imgLabel / 255.0
    imgAlbedo_normalized = imgLabel_normalized / (imgShading_normalized + .0001)
    imgDepth_normalized = imgDepth / 255.0

    INPUT[p[k], :, :, :] = imgLabel_normalized
    TARGET[p[k], :, :, :] = imgData_normalized
    DEPTH[p[k], :, :, :] = imgDepth_normalized
    ALBEDO[p[k], :, :, :] = imgAlbedo_normalized
    SHADING[p[k], :, :, :] = imgShading_normalized

    k = k + 1

    print(str(k) + '-INPUT' + str(INPUT.shape) + '-TARGET' + str(TARGET.shape))
    sys.stdout.flush()  # ?

dset_input = h5fw.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
INPUT = None
print('>>>>INPUT file generated')
dset_depth = h5fw.create_dataset(name='DEPTH', shape=DEPTH.shape, data=DEPTH, dtype=np.float32)
dset_target = h5fw.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
dset_albedo = h5fw.create_dataset(name='ALBEDO', shape=ALBEDO.shape, data=ALBEDO, dtype=np.float32)
dset_shading = h5fw.create_dataset(name='SHADING', shape=SHADING.shape, data=SHADING, dtype=np.float32)
print('>>>>TARGET file generated')
print('>>>>save file' + 'training' + 'INPUT_' + str(SIZE_INPUT) + 'TARGET_' + str(SIZE_TARGET))
h5fw.close()

h5fw = h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_3' + '.h5'), 'w')
ALBEDO = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
SHADING = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
INPUT = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
TARGET = np.empty(shape=(total, SIZE_TARGET, SIZE_TARGET, 3))
DEPTH = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 1))
k = 0
p = np.random.permutation(total)
# print(p)
# names = glob.glob(LABEL_PATH + '*.png')[total:total*2]
for data_image in names[2*total:3*total]:
    string_label = path_leaf(data_image)
    string_shading = 's' + string_label[3:]
    string_data = 'inp' + string_label[3:]
    string_depth = 'depth' + string_label[3:]
    # print(string_data)
    print(string_label)
    shading_image_name = Shading_PATH + string_shading
    data_image_name = DATA_PATH + string_data
    depth_image_name = DATA_PATH + string_depth
    # BI_img_name = BI_PATH + HR_img_name[12:19] + '.png'
    # print(label_image_name)
    imgShading = cv2.imread(shading_image_name)
    imgData = cv2.imread(data_image_name)
    imgLabel = cv2.imread(data_image)
    imgDepth = cv2.imread(depth_image_name)[:, :, 0:1]
    # normalizing the input and target images
    imgShading_normalized = cv2.cvtColor(imgShading, cv2.COLOR_BGR2RGB) / 255.0
    imgData_normalized = imgData / 255.0
    imgLabel_normalized = imgLabel / 255.0
    imgAlbedo_normalized = imgLabel_normalized / (imgShading_normalized + .0001)
    imgDepth_normalized = imgDepth / 255.0

    INPUT[p[k], :, :, :] = imgLabel_normalized
    TARGET[p[k], :, :, :] = imgData_normalized
    DEPTH[p[k], :, :, :] = imgDepth_normalized
    ALBEDO[p[k], :, :, :] = imgAlbedo_normalized
    SHADING[p[k], :, :, :] = imgShading_normalized

    k = k + 1

    print(str(k) + '-INPUT' + str(INPUT.shape) + '-TARGET' + str(TARGET.shape))
    sys.stdout.flush() 
dset_input = h5fw.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
INPUT = None
print('>>>>INPUT file generated')
dset_depth = h5fw.create_dataset(name='DEPTH', shape=DEPTH.shape, data=DEPTH, dtype=np.float32)
dset_target = h5fw.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
dset_albedo = h5fw.create_dataset(name='ALBEDO', shape=ALBEDO.shape, data=ALBEDO, dtype=np.float32)
dset_shading = h5fw.create_dataset(name='SHADING', shape=SHADING.shape, data=SHADING, dtype=np.float32)
print('>>>>TARGET file generated')
print('>>>>save file' + 'training' + 'INPUT_' + str(SIZE_INPUT) + 'TARGET_' + str(SIZE_TARGET))
h5fw.close()


h5fw = h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training_albedo_4' + '.h5'), 'w')
ALBEDO = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
SHADING = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
INPUT = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
TARGET = np.empty(shape=(total, SIZE_TARGET, SIZE_TARGET, 3))
DEPTH = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 1))
k = 0
p = np.random.permutation(total)
# print(p)
# names = glob.glob(LABEL_PATH + '*.png')[total:total*2]
for data_image in names[3*total:4*total]:
    string_label = path_leaf(data_image)
    string_shading = 's' + string_label[3:]
    string_data = 'inp' + string_label[3:]
    string_depth = 'depth' + string_label[3:]
    # print(string_data)
    print(string_label)
    shading_image_name = Shading_PATH + string_shading
    data_image_name = DATA_PATH + string_data
    depth_image_name = DATA_PATH + string_depth
    # BI_img_name = BI_PATH + HR_img_name[12:19] + '.png'
    # print(label_image_name)
    imgShading = cv2.imread(shading_image_name)
    imgData = cv2.imread(data_image_name)
    imgLabel = cv2.imread(data_image)
    imgDepth = cv2.imread(depth_image_name)[:, :, 0:1]
    # normalizing the input and target images
    imgShading_normalized = cv2.cvtColor(imgShading, cv2.COLOR_BGR2RGB) / 255.0
    imgData_normalized = imgData / 255.0
    imgLabel_normalized = imgLabel / 255.0
    imgAlbedo_normalized = imgLabel_normalized / (imgShading_normalized + .0001)
    imgDepth_normalized = imgDepth / 255.0

    INPUT[p[k], :, :, :] = imgLabel_normalized
    TARGET[p[k], :, :, :] = imgData_normalized
    DEPTH[p[k], :, :, :] = imgDepth_normalized
    ALBEDO[p[k], :, :, :] = imgAlbedo_normalized
    SHADING[p[k], :, :, :] = imgShading_normalized

    k = k + 1

    print(str(k) + '-INPUT' + str(INPUT.shape) + '-TARGET' + str(TARGET.shape))
    sys.stdout.flush()  # ?

dset_input = h5fw.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
INPUT = None
print('>>>>INPUT file generated')
dset_depth = h5fw.create_dataset(name='DEPTH', shape=DEPTH.shape, data=DEPTH, dtype=np.float32)
dset_target = h5fw.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
dset_albedo = h5fw.create_dataset(name='ALBEDO', shape=ALBEDO.shape, data=ALBEDO, dtype=np.float32)
dset_shading = h5fw.create_dataset(name='SHADING', shape=SHADING.shape, data=SHADING, dtype=np.float32)
print('>>>>TARGET file generated')
print('>>>>save file' + 'training' + 'INPUT_' + str(SIZE_INPUT) + 'TARGET_' + str(SIZE_TARGET))
h5fw.close()
