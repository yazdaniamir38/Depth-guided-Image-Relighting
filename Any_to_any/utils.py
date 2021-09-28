import numpy as np
import math, os, torch, sys
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
import importlib
import logging
import scipy.io as sio
path='./dataset_generation/pre_reqs/'
def checkdirctexist(dirct):
    if not os.path.exists(dirct):
        os.makedirs(dirct)


def PSNR_self(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred -gt)
    rmse = math.sqrt(np.mean(imdff.cpu().data[0].numpy() ** 2))
    if rmse == 0:
        return 100
    return 20.0 * math.log10(1.0/rmse)


# def save_checkpoint(model, epoch, save_path):
#     model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
#     state = {"epoch": epoch, "model": model}
#     # check path status
#     if not os.path.exists("model/"):
#         os.makedirs("model/")
#     # save model
#     torch.save(state, model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))


def save_checkpoint(model, epoch, opt, save_path,suffix=''):
    model_out_path = os.path.join(save_path, suffix+"model_epoch_{}.pth".format(epoch))
    if opt.cuda:
        model_file = model.module.__class__.__module__
        model_class = model.module.__class__.__name__
        model_state = model.module.state_dict()
    else:
        model_file = model.__class__.__module__
        model_class = model.__class__.__name__
        model_state = model.state_dict()

    state = {"epoch": epoch,
             "model_state": model_state,
             "opt": opt,
             "args": sys.argv,
             "model_file": model_file,
             "model_class": model_class}

    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")

    # save model
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def load_checkpoint(path,name='', **extra_params):
    if 'iscuda' in extra_params:
        iscuda = extra_params.pop('iscuda')
    else:
        iscuda = True
    if iscuda:
        dict = torch.load(path, map_location='cuda')
    else:
        dict = torch.load(path, map_location='cpu')
    print(dict["model_file"])
    print(dict["model_class"])
    if name:
      md = importlib.import_module(name)
    else:
        md = importlib.import_module(dict['model_file'])
    # if extra_params:
    #     model = eval("md.{}".format(dict["model_class"]))(**extra_params)
    # else:
    model = eval("md.{}".format(dict["model_class"]))()
    model.load_state_dict(dict["model_state"])
    return model, dict["epoch"], dict

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def compute_normals(image_name):
    _, filename = os.path.split(image_name)
    normals=sio.loadmat(path+filename[:-4]+'normal.mat')['normal']
    normals = torch.from_numpy(normals)
    normals = normals.permute([2, 0, 1]).unsqueeze(0).float()
    return normals
