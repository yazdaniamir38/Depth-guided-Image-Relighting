import argparse, os, time
os.environ['TORCH_HOME'] = '/cvdata2/amir/torch/'
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import HS_multiscale_DSet
from dataset import DatasetFromHdf5
import utils
import kornia
import rgb_to_lab
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from Vgg16 import Vgg16
import pytorch_msssim
import importlib
import common
from dense121_classifier import dense121
import glob
import ntpath
import cv2
import PIL
from PIL import Image
import skimage
import skimage.measure
import skimage.io as io
import torchvision.transforms as transforms
# Training settings
parser = argparse.ArgumentParser(description="Pytorch DRRN")
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--traindata", default="../new_names/", type=str, help="Training datapath")#SynthesizedfromN18_256s64

parser.add_argument("--nEpochs", type=int, default=150, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--gpu_ids", default='1', type=str, help='ID for gpus')
parser.add_argument("--aug", action="store_true", help="Use aug?")

# parser.add_argument("--resume", default="model/dense_residual_deepModel_AT_actual_finetune/model_epoch_4.pth", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--resume", default="model/dense169AtJw_SE_separate_bottlenecks_dilation_inception_multi_scale_shading_anytoany_multi_scale/model_epoch_20.pth", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="shading_anytoany_multi_scale", type=str, help='ID for training')
parser.add_argument("--model", default="dense169AtJw_SE_separate_bottlenecks_dilation_inception_multi_scale", type=str, help="unet or drrn or runet")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    save_path = os.path.join('.', "model", "{}_{}".format(opt.model, opt.ID),'val_vgg')
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
        mod = importlib.import_module('dense169AtJw_SE_separate_bottlenecks_no_dilation_inception_multi_scale')
        try:
            model = mod.Model()
        except AttributeError:
            model = mod.Dense()
    except FileExistsError:
        raise SyntaxError('wrong model type {}'.format(opt.model))


    # model.freeze('encoder')

    print("===> Loading datasets")
    # train_set = DatasetFromHdf5(os.walk('../h5/'), opt.patchSize, opt.aug)
    train_set = HS_multiscale_DSet(opt.traindata)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    criterion = nn.MSELoss()
    Absloss = nn.L1Loss()
    # ssim_loss = pytorch_msssim.MSSSIM()
    ssim_loss=kornia.losses.SSIM(11)
    psnr_loss=kornia.losses.PSNRLoss(1)
    #loss_var = torch.std()

    vgg = Vgg16(requires_grad=False)
    light_estimation=dense121()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint: {}".format(opt.resume))
            model, epoch, dict = utils.load_checkpoint(opt.resume,file_name='dense169AtJw_SE_separate_bottlenecks_no_dilation_inception_multi_scale')
            opt.start_epoch = epoch + 1
        else:
            raise FileNotFoundError("===> no checkpoint found at {}".format(opt.resume))

    print("===> Setting GPU")
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        model = torch.nn.DataParallel(model).cuda()
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.transpose(J_direct[10, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.transpose(target[10, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
        # plt.show()
        criterion2=nn.CrossEntropyLoss().cuda()
        criterion = criterion.cuda()
        Absloss = Absloss.cuda()
        ssim_loss = ssim_loss.cuda()
        #loss_var = loss_var.cuda()
        vgg = vgg.cuda()
        light_estimation=light_estimation.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)#weight_decay=opt.weight_decay
    log_dir = os.path.join(save_path, 'train.log')
    logger = utils.initialize_logger(log_dir)
    print("===> Training")
    # best1=.58
    best=18+6.7
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # utils.save_checkpoint(model, epoch, opt, save_path)
        # eval(model, '../validation/', '../AIM2020_track3_validation_gt_upsampled/', epoch, logger, opt)
        train(training_data_loader, optimizer, model, criterion,criterion2, Absloss, ssim_loss, psnr_loss,epoch, light_estimation,vgg)
        bench1,bench2=eval(model, '../validation/', '../AIM2020_track3_validation_gt_upsampled/', epoch, logger,384, opt)
        if 10*bench1+bench2>best:
            utils.save_checkpoint(model, epoch, opt, save_path)
            best=10*bench1+bench2
            # best2=bench2
        if epoch%2==0:
            # eval(model, '../validation/', '../AIM2020_track3_validation_gt_upsampled/', epoch, logger, opt)
            utils.save_checkpoint(model, epoch, opt, save_path)
        # os.system("python eval.py --cuda --model=model/model_epoch_{}.pth".format(epoch))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch  // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion,criterion2, Absloss, ssim_loss,psnr_loss, epoch, light_estimation,vgg):

    writer = SummaryWriter(log_dir='./records/{}_{}/'.format(opt.model, opt.ID))

    # lr policy
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    total_iter = 0
    for iteration, batch in enumerate(training_data_loader,1):
        input, depth1, target, shading ,albedo,guide,depth2= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]),Variable(batch[4]),Variable(batch[5]),Variable(batch[6])
        if opt.cuda:
            input = input.cuda()
            depth1 = depth1.cuda()
            target = target.cuda()
            shading = shading.cuda()
            albedo=albedo.cuda()
            guide=guide.cuda()
            depth2=depth2.cuda()
        # outputs
        J_total, J_direct,J_AT,A,T,w,_ = model(input, depth1,guide,depth2, opt)
        # plt.imshow(input[0,:,:,:].cpu())
        # plt.show()
        # features_target = vgg(target)
        # features_output_eh = vgg(J_total)
        # haze = target*T + (1.0 - T)*A
        loss_mse_total = criterion(J_total, target)
        loss_mse_direct = criterion(J_direct, target)
        loss_mse_AT = criterion(J_AT, target)
        loss_mse_S = criterion(T, shading)
        loss_mse_A=criterion(A,albedo)

        # light_features_target=light_estimation(target)
        # light_features_guide=light_estimation(guide)
        # light_features_output=light_estimation(J_total)
        # light_error=criterion(light_features_output.relu3_3,light_features_guide.relu3_3)+criterion(light_features_output.relu4_3,light_features_guide.relu4_3)+criterion2(light_features_output.color,torch.argmax(light_features_guide.color,1))+criterion2(light_features_output.dir,torch.argmax(light_features_guide.dir,1))+criterion(light_features_output.relu1_2,light_features_target.relu1_2)+criterion(light_features_output.relu2_2,light_features_target.relu2_2)
        # lab_tar=rgb_to_lab.rgb_to_lab(kornia.color.bgr_to_rgb(target))
        # lab_J=rgb_to_lab.rgb_to_lab(kornia.color.bgr_to_rgb(J_total))
        # lab_tar = rgb_to_lab.rgb_to_lab(target)
        # lab_J=rgb_to_lab.rgb_to_lab(J_total)
        # loss_mse_recon = criterion(haze, input)
        # loss_l1_total = Absloss(J_total, target)
        # loss_l1_direct = Absloss(J_direct, target)
        # loss_l1_AT = Absloss(J_AT, target)
        features_output_eh=vgg(J_total)
        features_target=vgg(target)
        loss_vgg = criterion(features_output_eh.relu2_2, features_target.relu2_2) + criterion(
            features_output_eh.relu1_2, features_target.relu1_2) + criterion(features_output_eh.relu3_3,
                                                                             features_target.relu3_3)
        # loss_vgg = loss_vgg / 3.0
        # s_loss = 1.0 - ssim_loss(target, J_total)
        s_loss = torch.mean(ssim_loss(target,J_total))
        PSNR=60-psnr_loss(target,J_total)
        # lab_loss=Absloss(lab_J,lab_tar)
        # var_loss = torch.std(A)**2
        # loss=2*loss_mse_direct+s_loss+(1/300)*PSNR+.02*light_error
        # loss = 4*loss_mse_total +.7*(loss_mse_direct + loss_mse_AT)+ .5*loss_mse_S + 1.5*s_loss +.1*loss_mse_A+.003*PSNR+.003*light_error #+.00005*lab_loss# + .01*var_loss++ 0.05 * loss_vgg
        loss =  1.5*loss_mse_total + .5 * (loss_mse_direct + loss_mse_AT) +  .5*loss_mse_S + .85*s_loss + .2 * loss_mse_A + .001 * PSNR + .025 * loss_vgg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer.add_scalar('data/scalar1', loss.item(), total_iter)

        if iteration % 50 == 0:
            psnr_run_total = utils.PSNR_self(J_total.clone(), target.clone())
            psnr_run_direct = utils.PSNR_self(J_direct.clone(), target.clone())
            psnr_run_AT = utils.PSNR_self(J_AT.clone(), target.clone())

            # input_display = vutils.make_grid(input, normalize=False, scale_each=True)
            # writer.add_image('Image/train_input', input_display, total_iter)

            # J_total_display = vutils.make_grid(J_total, normalize=False, scale_each=True)
            # J_direct_display = vutils.make_grid(J_direct, normalize=False, scale_each=True)
            # J_AT_display = vutils.make_grid(J_AT, normalize=False, scale_each=True)
            # shading_display = vutils.make_grid(shading, normalize=False, scale_each=True)
            # T_display = vutils.make_grid(T, normalize=False, scale_each=True)
            # A_display=vutils.make_grid(A, normalize=False, scale_each=True)
            # writer.add_image('Image/train_output', J_total_display, total_iter)
            # writer.add_image('Image/train_output', J_AT_display, total_iter)
            # writer.add_image('Image/train_output', J_direct_display, total_iter)
            # writer.add_image('Image/train_output', shading_display, total_iter)
            # writer.add_image('Image/train_output', T_display, total_iter)
            # writer.add_image('Image/train_output', A_display, total_iter)

            # gt_display = vutils.make_grid(target, normalize=False, scale_each=True)
            # writer.add_image('Image/train_target', gt_display, total_iter)

            # psnr_run = 100
            print(
                "===> Epoch[{}]({}/{}): Loss_total: {:.10f},Loss_ssim: {:.10f}, Loss_mse_total: {:.10f},loss_mse_direct:{:.10f},loss_mse_AT: {:.10f},psnr_total: {:.3f},psnr_direct: {:.3f},psnr_AT: {:.3f}"
                "".format(epoch, iteration, len(training_data_loader),loss.data.item(), s_loss.data.item(),loss_mse_total.data.item(),loss_mse_direct.data.item(),loss_mse_AT.data.item(),psnr_run_total,
                           psnr_run_direct,psnr_run_AT))
            # print(
            # "===> Epoch[{}]({}/{}): Loss_mse: {:.10f} ,psnr: {:.3f}"
            # "".format(epoch, iteration, len(training_data_loader), loss_mse_direct.data.item(),psnr_run_direct))
        total_iter += 1
# def eval(model,val_path,val_gt_path,epoch,logger,opt):
#     names=glob.glob(val_path+'/input/*.png')
#     s=0
#     PSNR=0
#     with torch.no_grad():
#         model = model.eval()
#         model = model.train(mode=False)
#         for inp_name in names:
#             _,img_name = ntpath.split(inp_name)
#             gt_name=val_gt_path+img_name
#             depth1_name=inp_name[:-3]+'npy'
#             guide_name=val_path+'/guide/'+img_name
#             depth2_name=guide_name[:-3]+'npy'
#             # img=Variable(kornia.color.bgr_to_rgb(torch.from_numpy((np.rollaxis(cv2.imread(inp_name)/255,2))[np.newaxis,:,:,:]).float()).cuda())
#             img=torch.from_numpy(np.rollaxis(io.imread(inp_name)[:,:,0:3]/255,2)[np.newaxis,:,:,:]).float().cuda()
#             guide=torch.from_numpy(np.rollaxis(io.imread(guide_name)[:,:,0:3]/255,2)[np.newaxis,:,:,:]).float().cuda()
#             # gt=kornia.color.bgr_to_rgb(torch.from_numpy((np.rollaxis(cv2.imread(gt_name)/255,2))[np.newaxis,:,:,:])).float()
#             gt=io.imread(gt_name)[:,:,0:3]
#             depth1=Variable(torch.from_numpy((np.load(depth1_name,allow_pickle=True)).item().get('normalized_depth')).float().unsqueeze(0).unsqueeze(0).cuda())
#             depth2 = Variable(torch.from_numpy(
#                 (np.load(depth2_name, allow_pickle=True)).item().get('normalized_depth')).float().unsqueeze(
#                 0).unsqueeze(0).cuda())
#             pred,_,_,_,_,_,_=model(img,depth1,guide,depth2,opt)
#             pred=np.transpose(np.asarray(pred.detach().cpu().squeeze(),dtype=float),[1,2,0])
#             pred*=255.
#             pred[pred<0]=0
#             pred[pred>255]=255
#             pred = (pred).astype('uint8')
#             s+=skimage.measure.compare_ssim(gt, pred, multichannel=True)
#             PSNR+=skimage.measure.compare_psnr(gt, pred)
#             io.imsave('results/dense121AtJw_SE_separate_bottlenecks_dilation_inception_shading_anytoany_no_freeze_crop/model_epoch_7/'+img_name,pred)
#     print("===> Validation at Epoch[{}] ssim: {:.10f}, PSNR: {:.10f} "
#           "".format(epoch,s/len(names),PSNR/len(names)))
#     logger.info("Epoch [%02d], SSIM:%.9f, PSNR: %.9f"
#                 % (epoch,s/len(names),PSNR/len(names)))
#     return s/len(names),PSNR/len(names)
def eval(model,val_path,val_gt_path,epoch,logger,size,opt):
    names=glob.glob(val_path+'/input/*.png')
    s=0
    PSNR=0
    To_tensor=transforms.ToTensor()
    with torch.no_grad():
        model = model.eval()
        model = model.train(mode=False)
        for inp_name in names:
            _,img_name = ntpath.split(inp_name)
            gt_name=val_gt_path+img_name
            depth1_name=inp_name[:-3]+'npy'
            guide_name=val_path+'/guide/'+img_name
            depth2_name=guide_name[:-3]+'npy'
            img=Image.fromarray(io.imread(inp_name)[:,:,0:3])
            img=To_tensor(img.resize([size,size])).float().cuda().unsqueeze(0)
            guide=Image.fromarray(io.imread(guide_name)[:,:,0:3])
            guide=To_tensor(guide.resize([size,size])).float().cuda().unsqueeze(0)
            depth1 = Image.fromarray(np.load(depth1_name, allow_pickle=True).item().get('normalized_depth'))
            depth1 = To_tensor(depth1.resize([size, size])).float().cuda().unsqueeze(0)
            depth2=Image.fromarray(np.load(depth2_name, allow_pickle=True).item().get('normalized_depth'))
            depth2=To_tensor(depth2.resize([size,size])).float().cuda().unsqueeze(0)
            gt=io.imread(gt_name)[:,:,0:3]
            # gt=Image.fromarray(io.imread(gt_name)[:,:,0:3])
            # gt=gt.resize([384,384])
            # gt=np.asarray(gt).astype('uint8')
            pred,_,_,_,_,_,_=model(img,depth1,guide,depth2,opt)
            pred=F.interpolate(pred,scale_factor=2.667,mode='bicubic')
            # pred=F.interpolate(pred,scale_factor=4,mode='bicubic')
            pred=np.transpose(np.asarray(pred.detach().cpu().squeeze(),dtype=float),[1,2,0])
            # pred=Image.fromarray(pred)
            # pred=np.array(pred.resize([1024,1024]))
            pred*=255.
            pred[pred<0]=0
            pred[pred>255]=255
            pred = (pred).astype('uint8')
            s+=skimage.measure.compare_ssim(gt, pred, multichannel=True)
            PSNR+=skimage.measure.compare_psnr(gt, pred)
            # pred=cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)
            # cv2.imwrite('results/dense121AtJw_SE_separate_bottlenecks_dilation_inception_shading_anytoany_unfreezed_together/model_epoch_40_opencv/'+img_name,pred)
            # utils.checkdirctexist('results/dense169AtJw_SE_separate_bottlenecks_no_dilation_inception_shading_anytoany_multi_scale/model_epoch_19/')
            # io.imsave('results/dense169AtJw_SE_separate_bottlenecks_no_dilation_inception_shading_anytoany_multi_scale/model_epoch_19/'+img_name,pred)
            # utils.checkdirctexist('../validation_256/')
            # io.imsave('../validation_256/' + img_name,gt)
    print("===> Validation at Epoch[{}] ssim: {:.10f}, PSNR: {:.10f} "
          "".format(epoch,s/len(names),PSNR/len(names)))
    logger.info("Epoch [%02d], SSIM:%.9f, PSNR: %.9f"
                % (epoch,s/len(names),PSNR/len(names)))
    return s/len(names),PSNR/len(names)

# def eval(model,val_path,val_gt_path,epoch,logger,opt):
#     names=glob.glob(val_path+'/input/*.png')
#     s=0
#     PSNR=0
#     To_tensor=transforms.ToTensor()
#     with torch.no_grad():
#         model = model.eval()
#         model = model.train(mode=False)
#         for inp_name in names:
#             _,img_name = ntpath.split(inp_name)
#             gt_name=val_gt_path+img_name
#             depth1_name=inp_name[:-3]+'npy'
#             guide_name=val_path+'/guide/'+img_name
#             depth2_name=guide_name[:-3]+'npy'
#     #         # img=Variable(kornia.color.bgr_to_rgb(torch.from_numpy((np.rollaxis(cv2.imread(inp_name)/255,2))[np.newaxis,:,:,:]).float()).cuda())
#             img=torch.from_numpy(np.rollaxis(io.imread(inp_name)[:,:,0:3]/255,2)[np.newaxis,:,:,:]).float().cuda()
#     #         # guide=torch.from_numpy(np.rollaxis(io.imread(guide_name)[:,:,0:3]/255,2)[np.newaxis,:,:,:]).float().cuda()
#             guide=Image.fromarray(io.imread(guide_name)[:,:,0:3])
#             guide=To_tensor(guide.resize([256,256])).float().cuda().unsqueeze(0)
#             depth1 = Variable(torch.from_numpy((np.load(depth1_name, allow_pickle=True)).item().get('normalized_depth')).float().unsqueeze(0).unsqueeze(0).cuda())
#             depth2=Image.fromarray(np.load(depth2_name, allow_pickle=True).item().get('normalized_depth'))
#             depth2=To_tensor(depth2.resize([256,256])).float().cuda().unsqueeze(0)
#             # depth2 = Variable(torch.from_numpy(
#             #     (np.load(depth2_name, allow_pickle=True)).item().get('normalized_depth')).float().unsqueeze(
#     #         #     0).unsqueeze(0).cuda())
#             pred=torch.empty([1,3,1024,1024])
#             for h in range(4):
#                 inp=torch.empty([4,3,256,256])
#                 d_inp=torch.empty([4,1,256,256])
#                 for w in range(4):
#                     inp[w,:,:,:]=img[0,:,h*256:(h+1)*256,w*256:(w+1)*256]
#                     d_inp[w,:,:,:]=depth1[0,:,h*256:(h+1)*256,w*256:(w+1)*256]
#
#                 out,_,_,_,_,_,_=model(inp.cuda(),d_inp.cuda(),torch.cat([guide,guide,guide,guide],0),torch.cat([depth2,depth2,depth2,depth2],0),opt)
#                 out=out*torch.where(d_inp.cuda()>0,torch.tensor([1]).cuda(),torch.tensor([0]).cuda())
#                 for i in range(4):
#                     pred[:,:,h*256:(h+1)*256,i*256:(i+1)*256]=out[i,:,:,:]
#
#             # gt=kornia.color.bgr_to_rgb(torch.from_numpy((np.rollaxis(cv2.imread(gt_name)/255,2))[np.newaxis,:,:,:])).float()
#             gt=io.imread(gt_name)[:,:,0:3]
#
#             # pred,_,_,_,_,_,_=model(img,depth1,guide,depth2,opt)
#             pred=np.transpose(np.asarray(pred.detach().cpu().squeeze(),dtype=float),[1,2,0])
#             pred*=255.
#             pred[pred<0]=0
#             pred[pred>255]=255
#             pred = (pred).astype('uint8')
#             s+=skimage.measure.compare_ssim(gt, pred, multichannel=True)
#             PSNR+=skimage.measure.compare_psnr(gt, pred)
#             io.imsave('results/dense121AtJw_SE_separate_bottlenecks_dilation_inception_shading_anytoany_no_freeze_crop/model_epoch_7/'+img_name,pred)
#     print("===> Validation at Epoch[{}] ssim: {:.10f}, PSNR: {:.10f} "
#           "".format(epoch,s/len(names),PSNR/len(names)))
#     logger.info("Epoch [%02d], SSIM:%.9f, PSNR: %.9f"
#                 % (epoch,s/len(names),PSNR/len(names)))
#     return s/len(names),PSNR/len(names)

if __name__ == "__main__":
    main()
