import numpy as np
import matplotlib.pyplot as plt
import glob
import skimage.io as io
import scipy.io as sio
import ntpath

files=glob.glob('../validation/'+'*.npy')

for name in files:
    _,file_name=ntpath.split(name)
    depth = np.load(name, allow_pickle=True)
    depth=depth.item().get('normalized_depth')
    depth=np.asarray(depth*255,dtype=float)
    h,w=np.shape(depth)
    normals=np.empty((h,w,3),dtype=float)
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
            normals[i,j,:]=d

    sio.savemat('./pre_reqs/'+file_name[:-4]+'normal'+'.mat',{'normal':normals})

