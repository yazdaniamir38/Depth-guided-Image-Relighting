import numpy as np
import scipy.io as io
import glob, os
DATA_PATH='../input/'
for data_image in glob.glob(DATA_PATH + '*.npy'):
    # string_data = path_leaf(data_image)
    data=np.load(data_image,allow_pickle=True)
    data.item().get('normalized_depth')
    io.savemat(data_image[:-3]+'mat',{'depth':data})