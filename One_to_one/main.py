import argparse, os, time
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DatasetFromHdf5
import utils
import kornia
import rgb_to_lab
import torchvision.utils as vutils
from Vgg16 import Vgg16
import importlib
import common
from dense121 import dense121
from le import le
import glob
import ntpath
import cv2
import skimage
import skimage.measure
import skimage.io as io
import scipy.io as sio
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
# Training settings
parser = argparse.ArgumentParser(description="Pytorch DRRN")
parser.add_argument("--batchSize", type=int, default=6, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--traindata", default="./h5/data/", type=str, help="Training datapath")#SynthesizedfromN18_256s64

parser.add_argument("--nEpochs", type=int, default=150, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--gpu_ids", default="1", type=str, help='ID for gpus')
parser.add_argument("--aug", action="store_true", help="Use aug?")


parser.add_argument("--resume", default="model/", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="shading_lighting_estimation", type=str, help='ID for training')
parser.add_argument("--model", default="dense121", type=str, help="unet or drrn or runet")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    save_path = os.path.join('.', "model", "{}_{}".format(opt.model, opt.ID),'val')
    log_dir = './records/{}_{}/'.format(opt.model, opt.ID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    cuda = opt.cuda
    if cuda  and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    # opt.seed = 4222

    print("Random Seed: ", opt.seed)

    cudnn.benchmark = True

    print("===> Building model")
    try:
        mod = importlib.import_module(opt.model)
        try:
            model = mod.Model()
        except AttributeError:
            model = mod.Dense()
    except FileExistsError:
        raise SyntaxError('wrong model type {}'.format(opt.model))


    print("===> Loading datasets")
    train_set = DatasetFromHdf5(os.walk(opt.traindata), opt.patchSize, opt.aug)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    criterion = nn.MSELoss()
    ssim_loss=kornia.losses.SSIM(11)
    psnr_loss=kornia.losses.PSNRLoss(1)
    light_estimation=dense121()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint: {}".format(opt.resume))
            model, epoch, dict = utils.load_checkpoint(opt.resume)
            opt.start_epoch = epoch + 1
        else:
            raise FileNotFoundError("===> no checkpoint found at {}".format(opt.resume))

    print("===> Setting GPU")
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        ssim_loss = ssim_loss.cuda()
        light_estimation=light_estimation.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)#weight_decay=opt.weight_decay
    log_dir = os.path.join(save_path, 'train.log')
    logger = utils.initialize_logger(log_dir)
    print("===> Training")
    best=17
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, psnr_loss,epoch, light_estimation,vgg)
        bench1,bench2= eval(model, './validation/', './validation_gt/', epoch, logger, opt)
        if 10*bench1+bench2 > best:
            utils.save_checkpoint(model, epoch, opt, save_path)
            best=10*bench1+bench2
        if epoch%5==0:

            utils.save_checkpoint(model, epoch, opt, save_path)
        # os.system("python eval.py --cuda --model=model/model_epoch_{}.pth".format(epoch))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch  // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss,psnr_loss, epoch, light_estimation,vgg):

    writer = SummaryWriter(log_dir='./records/{}_{}/'.format(opt.model, opt.ID))

    # lr policy
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    total_iter = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, depth, target, shading, albedo,normal= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(
            batch[3]), Variable(batch[4]),Variable(batch[5])
        if opt.cuda:
            input = input.cuda()
            depth = depth.cuda()
            target = target.cuda()
            shading = shading.cuda()
            albedo = albedo.cuda()
            normal=normal.cuda()
        # outputs
        _, I_direct, I_AS, A, S,W = model(input, depth,opt)
        I_total=W*J_direct+(1-W)*I_AS*normal
        loss_mse_total = criterion(I_total, target)
        loss_mse_direct = criterion(I_direct, target)
        loss_mse_AS = criterion(I_AS, target)
        loss_mse_S = criterion(S, shading)
        loss_mse_A = criterion(A, albedo)
        light_features_target = light_estimation(target)
        light_features_output = light_estimation(I_total)
        light_error = criterion(light_features_output.relu2_2, light_features_target.relu2_2) + criterion(
            light_features_output.relu1_2, light_features_target.relu1_2) + criterion(light_features_output.relu3_3,

        s_loss = torch.mean(ssim_loss(target, J_total))
        PSNR = 60 - psnr_loss(target, J_total)
        loss =  3*loss_mse_total +.6*(loss_mse_direct + loss_mse_AS) + .5*loss_mse_S + .8*s_loss + .2 * loss_mse_A + .001 * PSNR + .3 * light_error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer.add_scalar('data/scalar1', loss.item(), total_iter)

        if iteration % 50 == 0:
            psnr_run_total = utils.PSNR_self(I_total.clone(), target.clone())
            psnr_run_direct = utils.PSNR_self(I_direct.clone(), target.clone())
            psnr_run_AS = utils.PSNR_self(I_AS.clone(), target.clone())

            print(
                "===> Epoch[{}]({}/{}): Loss_total: {:.10f},Loss_ssim: {:.10f}, Loss_mse: {:.10f} , Loss_S: {:.10f} ,psnr: {:.3f}, psnr_AT: {:.3f}, psnr_direct: {:.3f}"
                "".format(epoch, iteration, len(training_data_loader),loss.data.item(), s_loss.data.item(), loss_mse_total.data.item(),
                          loss_mse_S.data.item(), psnr_run_total, psnr_run_AS, psnr_run_direct))
           
        total_iter += 1
def eval(model,val_path,val_gt_path,epoch,logger,opt,norm=''):
    names=glob.glob(val_path+'*.png')
    s=0
    PSNR=0
    to_tensor=transforms.ToTensor()
    with torch.no_grad():
        model = model.eval()
        model = model.train(mode=False)
        for inp_name in names:
            _,img_name = ntpath.split(inp_name)
            gt_name=val_gt_path+img_name
            depth_name=inp_name[:-3]+'npy'
            normal_name=inp_name[:-4]+'normal.mat'
            normal = sio.loadmat(normal_name)['normal']
            normal = torch.from_numpy(normal)
            normal = normal.permute([2, 0, 1]).unsqueeze(0).float().cuda()
            normal=F.interpolate(normal,scale_factor=.25)
            normal=(normal*-1+1)/2
            img=Image.fromarray(io.imread(inp_name)[:,:,0:3])
            img=img.resize([256,256])
            img=(to_tensor(img)).unsqueeze(0).float().cuda()
            if norm:
                img=norm(img)
            gt=io.imread(gt_name)[:,:,0:3]
            depth=Image.fromarray((np.load(depth_name,allow_pickle=True)).item().get('normalized_depth'))
            depth=depth.resize([256,256])
            depth=to_tensor(depth).unsqueeze(0).float().cuda()
            depth=depth/torch.max(depth)
            pred,I_dir,I_AS,_,_,W=model(img,depth,opt)
            pred=W*I_dir+(1-W)*I_AS*normal[:,1:2,:,:]
            pred = F.interpolate(pred, scale_factor=4,mode='bicubic')
            pred=np.transpose(np.asarray(pred.detach().cpu().squeeze(),dtype=float),[1,2,0])
            pred*=255.
            pred[pred<0]=0
            pred[pred>255]=255.
            pred = (pred).astype('uint8')
            s+=skimage.measure.compare_ssim(gt, pred, multichannel=True)
            PSNR+=skimage.measure.compare_psnr(gt, pred)
            
    print("===> Validation at Epoch[{}] ssim: {:.10f}, PSNR: {:.10f} "
          "".format(epoch,s/len(names),PSNR/len(names)))
    logger.info("Epoch [%02d], SSIM:%.9f, PSNR: %.9f"
                % (epoch,s/len(names),PSNR/len(names)))
    return s/len(names),PSNR/len(names)
if __name__ == "__main__":
    main()
