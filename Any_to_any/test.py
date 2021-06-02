import os
os.environ['TORCH_HOME'] = '/cvdata2/amir/torch/'
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2, os
import kornia
import PIL
from PIL import Image
# from tqdm import trange
import ntpath
import utils
import timeit
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import torchvision.transforms as transforms
import torch.nn.functional as F
parser = argparse.ArgumentParser(description="Pytorch AtJw(+D) Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", type=str, default="dense169_dilation", help="model path")
parser.add_argument("--model1", type=str, default="dense121", help="model path")
parser.add_argument("--model2", type=str, default="dense169", help="model path")
parser.add_argument("--dataset", default="./test_set/", type=str, help="dataset path")


def get_image_for_save(img):
    # img = img.data[0].numpy()
    # img=(img-np.amin(img))/(np.amax(img)-np.amin(img))
    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img

def shift_img(img, dirction=(0,0),dim=3):
    h1 = h2 = w1 = w2 = 0
    if not dirction[0]==0:
        if dirction[0] < 0:
            h1 = abs(dirction[0])
            h2 = 0
        elif dirction[0] > 0:
            h1 = 0
            h2 = dirction[0]

    if not dirction[1]==0:
        if dirction[1] < 0:
            w1 = abs(dirction[1])
            w2 = 0
        elif dirction[1] > 0:
            w1 = 0
            w2 = dirction[1]
    print(">>padding ... {}".format((h1, h2, w1, w2)))
    img = np.pad(img, ((h1, h2), (w1, w2), (0, 0)), 'reflect')
    return img

def crop_image_back(img, dirction=(0,0)):
    img = img.data[0].numpy()
    _, H, W = img.shape
    h1 = w1 = 0
    h2 = H
    w2 = W
    if not dirction[0] == 0:
        if dirction[0] < 0:
            h1 = abs(dirction[0])
            h2 = H
        elif dirction[0] > 0:
            h1 = 0
            h2 = -dirction[0]

    if not dirction[1] == 0:
        if dirction[1] < 0:
            w1 = abs(dirction[1])
            w2 = W
        elif dirction[1] > 0:
            w1 = 0
            w2 = -dirction[1]

    img = img[:, h1:h2, w1:w2]
    return img

def avg_imgs(imgs):
    number_of_imgs = len(imgs)
    img = sum(imgs) / np.float32(number_of_imgs)
    return img

opt = parser.parse_args()
cuda = opt.cuda
save_path = './results'

utils.checkdirctexist(save_path)

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model_path = os.path.join('model', "{}.pth".format(opt.model))
model_path1 = os.path.join('model', "{}.pth".format(opt.model1))
model_path2 = os.path.join('model', "{}.pth".format(opt.model2))
model, _,dict = utils.load_checkpoint(model_path,file_name='dense169_dilation', iscuda=opt.cuda)
model1, _,dict1 = utils.load_checkpoint(model_path1,file_name='dense121', iscuda=opt.cuda)
model2, _,dict2 = utils.load_checkpoint(model_path2,file_name='dense169', iscuda=opt.cuda)

if cuda:
    model = model.cuda()
    model1=model1.cuda()
    model2=model2.cuda()
image_list = glob.glob(os.path.join(opt.dataset,'input/','*.png'))

count = 0.0
with torch.no_grad():
    if cuda:
        model = model.cuda()
        model1 = model1.cuda()
        model2=model2.cuda()
    else:
        model = model.cpu()
        model1 = model1.cpu()
        model2=model2.cpu()
    model.eval()
    model1.eval()
    model2.eval()
    model = model.train(mode=False)
    model1 = model1.train(mode=False)
    model2=model2.train(mode=False)
    os.makedirs(save_path, exist_ok=True)

    start = timeit.default_timer()
    To_tensor = transforms.ToTensor()
    for image_name in image_list:
        count += 1
        _,name=ntpath.split(image_name)
        print("Processing ", image_name)
        og_depth1=np.load(image_name[:-3]+'npy',allow_pickle=True)
        og_depth1=Image.fromarray(og_depth1.item().get('normalized_depth'))
        depth_input1=og_depth1.resize([384,384])
        depth_input1=To_tensor(depth_input1).unsqueeze(0).float()
        depth_input1=depth_input1/torch.max(depth_input1)
        og_img = io.imread(image_name)[:,:,0:3]
        og_img=Image.fromarray(og_img)
        im_input=og_img.resize([384,384])
        im_input=To_tensor(im_input).unsqueeze(0).float()
        og_depth2=np.load(opt.dataset+'guide/'+name[:-3]+'npy',allow_pickle=True)
        og_depth2=og_depth2.item().get('normalized_depth')
        og_depth2=Image.fromarray(og_depth2)
        depth_input2=og_depth2.resize([384,384])
        depth_input2=To_tensor(depth_input2).unsqueeze(0).float()
        og_guide=io.imread(opt.dataset+'guide/'+name)[:,:,0:3]
        og_guide = Image.fromarray(og_guide)
        im_guide = og_guide.resize([384, 384])
        im_guide = To_tensor(im_guide).unsqueeze(0).float()
        og_H, og_W, og_C = [1024,1024,3]
        pads = [(0,0)]
        for pad in pads:
            if cuda:
                im_input = im_input.cuda()
                depth_input1=depth_input1.cuda()
                im_guide=im_guide.cuda()
                depth_input2=depth_input2.cuda()
            #model 1
            I_final, I_direct, I_int, A, S, w ,_= model(im_input,depth_input1,im_guide,depth_input2, opt)
            I_final=F.interpolate(I_final,scale_factor=1024/384,mode='bicubic')
            I_final = kornia.color.bgr_to_rgb(I_final)
            im_output_I_final = crop_image_back(I_final.cpu(), dirction=pad)
            #model 2
            I_final, I_direct, I_int, A, T, w, _ = model1(im_input, depth_input1, im_guide, depth_input2, opt)
            I_final = F.interpolate(I_final, scale_factor=1024/384,mode='bicubic')
            I_final = kornia.color.bgr_to_rgb(I_final)
            im_output_I_final = (im_output_I_final+crop_image_back(I_final.cpu(), dirction=pad))
            I_final, I_direct, I_int, A, T, w, _ = model2(im_input, depth_input1, im_guide, depth_input2, opt)
            I_final = F.interpolate(I_final, scale_factor=1024 / 384, mode='bicubic')
            I_final = kornia.color.bgr_to_rgb(I_final)
            im_output_I_final = (im_output_I_final + crop_image_back(I_final.cpu(), dirction=pad))/3

        im_output_forsave = get_image_for_save(im_output_I_final)
        path, filename = os.path.split(image_name)
        cv2.imwrite(os.path.join(save_path, "{}".format(filename)), im_output_forsave)

    stop = timeit.default_timer()
print("Save_path=", save_path)
print("It takes average {}s for processing".format((stop-start)/count))

