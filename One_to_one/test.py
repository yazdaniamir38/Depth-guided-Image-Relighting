import os
# os.environ['TORCH_HOME'] = '/cvdata/amir/torch/'
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import cv2, os
import kornia
import utils
import timeit
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
parser = argparse.ArgumentParser(description="Pytorch dense121 Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", type=str, default="dense121", help="model path")
parser.add_argument("--dataset", default="./test_set/", type=str, help="dataset path")
def get_image_for_save(img):
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

model_path = os.path.join('model',"{}.pth".format(opt.model))
model, _,dict = utils.load_checkpoint(model_path,name='dense121',iscuda=opt.cuda)
if cuda:
    model = model.cuda()
image_list = glob.glob(os.path.join(opt.dataset,'*.png'))

count = 0.0
with torch.no_grad():
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    start = timeit.default_timer()
    to_tensor=transforms.ToTensor()
    for image_name in image_list:
        count += 1
        print("Processing ", image_name)
        og_depth=np.load(image_name[:-3]+'npy',allow_pickle=True)
        og_depth=og_depth.item().get('normalized_depth')
        depth_input=torch.from_numpy(og_depth).unsqueeze(0).unsqueeze(0)
        depth_input=depth_input/torch.max(depth_input)
        depth_input1 = Image.fromarray(og_depth)
        depth_input1=depth_input1.resize([384,384])
        depth_input1 = to_tensor(depth_input1)
        depth_input1=depth_input1.unsqueeze(0)
        depth_input1 = depth_input1 / torch.max(depth_input1)
        normals=utils.compute_normals(image_name)
        normals1=F.interpolate(normals, scale_factor=384 / 1024,mode='bicubic')
        og_img = Image.fromarray(cv2.imread(image_name))
        im_input=to_tensor(og_img)
        im_input=im_input.unsqueeze(0)
        im_input1=og_img.resize([384,384])
        im_input1=to_tensor(im_input1)
        im_input1=im_input1.unsqueeze(0)
        og_H, og_W, og_C = [1024,1024,3]
        pads = [(0,0)]
        if cuda:
            im_input=im_input.cuda()
            im_input1=im_input1.cuda()
            depth_input=depth_input.cuda()
            depth_input1=depth_input1.cuda()
            normals=normals.cuda()
            normals1=normals1.cuda()
        for pad in pads:
            #1024x1024
            im_input=kornia.color.bgr_to_rgb(Variable(im_input.float()))
            I_final, I_direct, I_int, A, S, w = model(im_input, depth_input, opt)
            S = S* (normals[:, 1, :, :]*-1+1)/2
            I_final = w * I_direct + (1 - w) * A * S
            I_final = kornia.color.bgr_to_rgb(I_final)
            im_output_I_final = (crop_image_back(I_final.cpu(), dirction=pad))
            normals1 = (normals1 * -1 + 1) / 2
            #384x384
            im_input1 = kornia.color.bgr_to_rgb(Variable(im_input1.float()))
            I_final, I_direct, I_int, A, S, w = model(im_input1, depth_input1, opt)
            I_final = w * I_direct + (1 - w) * A * S* normals1[:,1:2,:,:]
            I_final = F.interpolate(I_final, scale_factor=1024 / 384, mode='bicubic',align_corners=True)
            I_final = kornia.color.bgr_to_rgb(I_final)
            #ensemble
            im_output_I_final = (.5*im_output_I_final+.5*crop_image_back(I_final.cpu(), dirction=pad))

        im_output_forsave = get_image_for_save(im_output_I_final)
        path, filename = os.path.split(image_name)
        cv2.imwrite(os.path.join(save_path, "{}".format(filename)), im_output_forsave)
    stop = timeit.default_timer()
print("Save_path=", save_path)
print("It takes average {}s for processing".format((stop-start)/count))

