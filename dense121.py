import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import common
import copy
import le
from network_module import *
import PixelUnShuffle
class multi_scale(nn.Module):
    def __init__(self,in_channels=3,start_channels=7,pad_type='zero',activation='lrelu',norm='none'):
        super(multi_scale,self).__init__()
        # Top subnetwork, K = 3
        self.top1 = Conv2dLayer(in_channels * (4 ** 3), start_channels * (2 ** 3), 3, 1, 1, pad_type=pad_type,
                                activation=activation, norm=norm)
        self.top21 = ResidualDenseBlock_5C(start_channels * (2 ** 3), start_channels * (2 ** 2), 3, 1, 1,
                                           pad_type=pad_type, activation=activation, norm=norm)
        self.top22 = GlobalBlock(start_channels * (2 ** 3), 3, 1, 1, pad_type=pad_type, activation=activation,
                                 norm=norm, sn=False, reduction=4)
        self.top3 = Conv2dLayer(start_channels * (2 ** 3), start_channels * (2 ** 3), 1, 1, 0, pad_type=pad_type,
                                activation=activation, norm=norm)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer(in_channels * (4 ** 2), start_channels * (2 ** 2), 3, 1, 1, pad_type=pad_type,
                                activation=activation, norm=norm)
        self.mid2 = Conv2dLayer(int(start_channels * (2 ** 2 + 2 ** 3 / 4)), start_channels * (2 ** 2), 3, 1, 1,
                                pad_type=pad_type, activation=activation, norm=norm)
        self.mid31 = ResidualDenseBlock_5C(start_channels * (2 ** 2), start_channels * (2 ** 1), 3, 1, 1,
                                           pad_type=pad_type, activation=activation, norm=norm)
        self.mid32 = GlobalBlock(start_channels * (2 ** 2), 3, 1, 1, pad_type=pad_type, activation=activation,
                                 norm=norm, sn=False, reduction=4)
        self.mid4 = Conv2dLayer(start_channels * (2 ** 2), start_channels * (2 ** 2), 3, 1, 1, pad_type=pad_type,
                                activation=activation, norm=norm)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer(in_channels * (4 ** 1), start_channels * (2 ** 1), 3, 1, 1, pad_type=pad_type,
                                activation=activation, norm=norm)
        self.bot2 = Conv2dLayer(int(start_channels * (2 ** 1 + 2 ** 2 / 4)), start_channels * (2 ** 1), 3, 1, 1,
                                pad_type=pad_type, activation=activation, norm=norm)
        self.bot31 = ResidualDenseBlock_5C(start_channels * (2 ** 1), start_channels * (2 ** 0), 3, 1, 1,
                                           pad_type=pad_type, activation=activation, norm=norm)
        self.bot32 = ResidualDenseBlock_5C(start_channels * (2 ** 1), start_channels * (2 ** 0), 3, 1, 1,
                                           pad_type=pad_type, activation=activation, norm=norm)
        self.bot33 = GlobalBlock(start_channels * (2 ** 1), 3, 1, 1, pad_type=pad_type, activation=activation,
                                 norm=norm, sn=False, reduction=4)
        self.bot4 = Conv2dLayer(start_channels * (2 ** 1), start_channels * (2 ** 1), 3, 1, 1, pad_type=pad_type,
                                activation=activation, norm=norm)
    def forward(self,x):
        x1 = PixelUnShuffle.pixel_unshuffle(x, 2)  # out: batch * 12 * 128 * 128
        x2 = PixelUnShuffle.pixel_unshuffle(x, 4)  # out: batch * 48 * 64 * 64
        x3 = PixelUnShuffle.pixel_unshuffle(x, 8)  # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)  # out: batch * 256 * 32 * 32
        x3 = self.top21(x3)  # out: batch * 256 * 32 * 32
        x3 = self.top22(x3)  # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)  # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)  # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)  # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)  # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)  # out: batch * 128 * 64 * 64
        x2 = self.mid31(x2)  # out: batch * 128 * 64 * 64
        x2 = self.mid32(x2)  # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)  # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)  # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)  # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)  # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)  # out: batch * 64 * 128 * 128
        x1 = self.bot31(x1)  # out: batch * 64 * 128 * 128
        x1 = self.bot32(x1)  # out: batch * 64 * 128 * 128
        x1 = self.bot33(x1)  # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)  # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)
        return x1
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DilationInception(nn.Module):
    def __init__(self, channel, nlayers=3, kernel_size=3, se_reduction=16):
        super(DilationInception, self).__init__()

        dilations = [i+1 for i in range(nlayers)]
        kernel_size_effective = [kernel_size + (kernel_size - 1) * (dilation - 1) for dilation in dilations]
        paddings = [(i - 1)//2 for i in kernel_size_effective]

        self.weights = nn.Parameter(0.1*torch.rand(nlayers), requires_grad=True)
        self.branches = nn.ModuleList()
        for dilation, padding in zip(dilations, paddings):
            self.branches.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=kernel_size, dilation=dilation, padding=padding),
            ))

        self.se = SELayer(channel, reduction=se_reduction)

    def forward(self, x):
        out = x
        for w, branch in zip(self.weights, self.branches):
            out = out + (w**2)*branch(x)

        return self.se(out)


class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2*32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3*32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4*32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5*32)
        self.relu6= nn.ReLU(inplace=True)
        self.bn7 = nn.BatchNorm2d(inter_planes)
        self.relu7= nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.conv6 = nn.Conv2d(in_planes + 5*32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        #out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)					   
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.interpolate(out, scale_factor=2, mode='bilinear')


class SEDenseDecoder(nn.Module):
    def __init__(self):
        super(SEDenseDecoder, self).__init__()
        ############# Block5-up  16-16 ##############
        self.se5 = SELayer(384+256)
        self.dense_block5 = BottleneckDecoderBlock(384+256, 256)
        self.trans_block5 = TransitionBlock(896, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.se6 = SELayer(256)
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        #Amir
        # self.se_refin = SELayer(19, 3)
        self.se_refin = SELayer(20, 3)
        self.conv_refin = nn.Conv2d(20, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = F.interpolate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2,g_lighting, x4,depth, opt):
        x42 = torch.cat([x4, x2,g_lighting], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(self.se5(x42)))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        x52 = torch.cat([x5, x1], 1)
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(self.se6(x52)))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        x8 = self.residual_block82(x8)
        x8 = torch.cat([x8, x,depth], 1)
        x9 = self.relu(self.conv_refin(self.se_refin(x8)))
        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out, mode='bilinear')
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out, mode='bilinear')
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out, mode='bilinear')
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out, mode='bilinear')
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.refine3(dehaze)
        return dehaze


class DenseDecoderDilationInception(nn.Module):
    def __init__(self):
        super(DenseDecoderDilationInception, self).__init__()
        ############# Block5-up  16-16 ##############
        self.di5 = DilationInception(384+256)
        self.dense_block5 = BottleneckDecoderBlock(384+256, 256)
        self.trans_block5 = TransitionBlock(896, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.di6 = DilationInception(256)
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)

        self.di9 = DilationInception(20, se_reduction=3)
        self.conv_refin = nn.Conv2d(20, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = F.interpolate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2,g_lighting, x4,depth, opt):
        x42 = self.di5(torch.cat([x4, x2,g_lighting], 1))
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        x52 = self.di6(torch.cat([x5, x1], 1))
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        x8 = self.residual_block82(x8)
        #Amir
        # x8 = torch.cat([x8, x], 1)
        x8 = torch.cat([x8, x,depth], 1)
        x9 = self.relu(self.conv_refin(self.di9(x8)))
        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out, mode='bilinear')
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out, mode='bilinear')
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out, mode='bilinear')
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out, mode='bilinear')
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.refine3(dehaze)
        return dehaze


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        ############# 256-256  ##############
        with torch.no_grad():
            haze_class = models.densenet121(pretrained=True)
            # haze_class.features.conv0
            # layer=haze_class.features.conv0.weight.clone().detach()
            haze_class.features.conv0=nn.Conv2d(8,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
            # haze_class.features.conv0.weight[:,0:3,:,:].copy_(layer)
            # haze_class.features.conv0.weight[:,4:7,:,:].copy_(layer)
            # haze_class.features.conv0.weight[:,3,:,:].copy_(layer[:,0,:,:]*0.0721+layer[:,1,:,:]*0.7154+layer[:,2,:,:]*0.2125)
            # haze_class.features.conv0.weight[:, 7, :, :].copy_(layer[:, 0, :, :] * 0.0721 + layer[:, 1, :, :] * 0.7154 + layer[:, 2, :, :] * 0.2125)

        # self.conv=haze_class.features.conv0
        # self.conv0=nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        # self.conv0.weight[:, 0:3, :, :]=copy.deepcopy(self.conv.weight)
        # self.conv0.weight[:, 3, :, :]=copy.deepcopy(self.conv.weight[:,0,:,:]*0.0721+self.conv.weight[:,1,:,:]*0.7154+self.conv.weight[:,2,:,:]*0.2125)
        # self.inp=nn.Conv2d(8,3,padding=2,kernel_size=(5,5),bias=False)
        self.multi_scale=multi_scale(in_channels=8,start_channels=16,pad_type='reflect')
        self.conv0 = haze_class.features.conv0
        # nn.init.xavier_normal(self.conv0.weight.data[:,3,:,:],gain=.02)
        # nn.init.xavier_normal(self.conv0.weight.data[:,7,:,:],gain=.02)
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0
        self.light_estimation=le.le()
        ############# Block1-down 64-64  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.Jdense_block4 = BottleneckDecoderBlock(512, 256)#512
        self.Jtrans_block4 = TransitionBlock(768, 128)#768
        self.Jresidual_block41 = ResidualBlock(128)
        self.Jresidual_block42 = ResidualBlock(128)
        self.ATdense_block4 = BottleneckDecoderBlock(512, 256)#512
        self.ATtrans_block4 = TransitionBlock(768, 128)#768
        self.ATresidual_block41 = ResidualBlock(128)
        self.ATresidual_block42 = ResidualBlock(128)
        self.Wdense_block4 = BottleneckDecoderBlock(512, 256)#512
        self.Wtrans_block4 = TransitionBlock(768, 128)#768
        self.Wresidual_block41 = ResidualBlock(128)
        self.Wresidual_block42 = ResidualBlock(128)

        self.decoder_A = SEDenseDecoder()
        self.decoder_T = SEDenseDecoder()
        # self.decoder_J = DenseDecoderDilationInception()
        self.decoder_J=SEDenseDecoder()
        self.decoder_w = SEDenseDecoder()
        self.sigA=nn.Sigmoid()
        self.convT1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.ResT = ResidualBlock(32)
        self.convT = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigT = nn.Sigmoid()

        self.convw1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.Resw = ResidualBlock(32)
        self.convw = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigw = nn.Sigmoid()

        # self.refine1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        # self.bn_refine1 = nn.BatchNorm2d(20)
        # self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        # self.bn_refine2 = nn.BatchNorm2d(20)
        # self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        # self.threshold = nn.Threshold(0.1, 0.1)
        # self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.bn_conv1010 = nn.BatchNorm2d(1)
        # self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.bn_conv1020 = nn.BatchNorm2d(1)
        # self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.bn_conv1030 = nn.BatchNorm2d(1)
        # self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.bn_conv1040 = nn.BatchNorm2d(1)
        # self.upsample = F.interpolate
        # self.relu = nn.ReLU(inplace=True)

        self.JGen = common.JGenerate()

    def split_params(self):
        pretrained_params = []
        rest_params = []
        for name, module in self.named_children():
            if (name == "conv0") or (name == "norm0") or (name == "relu0") or (name == "pool0") or \
                    (name == "dense_block1") or (name == "dense_block2") or (name == "dense_block3") or \
                    (name == "trans_block1") or (name == "trans_block2") or (name == "trans_block3"):
                for p in module.parameters():
                    pretrained_params.append(p)
            else:
                for p in module.parameters():
                    rest_params.append(p)

        return pretrained_params, rest_params

    def set_parameters(self, models, value):
        dicts = {
            'encoder': [self.conv0, self.norm0, self.relu0, self.pool0, self.dense_block1, self.dense_block2, self.dense_block3, self.trans_block1, self.trans_block2, self.trans_block3],
            'J_AT': [self.decoder_A, self.decoder_T, self.convT, self.convT1, self.sigT, self.ResT],
            'J_direct': [self.decoder_J], 
            'w': [self.decoder_w, self.convw1, self.convw, self.sigw, self.Resw]
        }
        if not isinstance(models, list):
            models = [models]

        for model in models:
            # print(model)
            for block in dicts[model]:
                # print(block.__class__.__name__)
                for module in block.modules():
                    for p in module.parameters():
                          p.requires_grad=value

    def freeze(self, models):
        print('Freezing the following:')
        print(models)
        self.set_parameters(models, False)

    def unfreeze(self, models):
        print('Unfreezing the following:')
        print(models)
        self.set_parameters(models, True)
    
    def forward(self, x,depth1,g,depth2 ,opt):
        ## 256x256
        # print("input {}".format((x!=x).any()))
        # inp=self.inp(torch.cat([x,depth1,g,depth2],1))
        x_multi_scale=self.multi_scale(torch.cat([x,depth1,g,depth2],1))
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x_multi_scale))))

        # print("x0 {}".format((x0!=x0).any()))
        ## 64 X 64
        x1 = self.dense_block1(x0)
        # print("x1 {}".format((x1!=x1).any()))
        # print x1.size()
        x1 = self.trans_block1(x1)

        # print("x1 {}".format((x1!=x1).any()))
        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()
        g_light=self.light_estimation(g).relu2_2
        # print("x2 {}".format((x2!=x2).any()))
        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))

        # print("x3 {}".format((x3!=x3).any()))
        # x3=Variable(x3.data,requires_grad=True)

        ## 8 X 8
        x4J = self.Jtrans_block4(self.Jdense_block4(x3))
        #Amir
        x4AT = self.ATtrans_block4(self.ATdense_block4(x3))
        x4W = self.Wtrans_block4(self.Wdense_block4(x3))
		
        x4J = self.Jresidual_block41(x4J)
        x4J = self.Jresidual_block42(x4J)
        #Amir
        x4AT = self.ATresidual_block41(x4AT)
        x4AT = self.ATresidual_block42(x4AT)
        x4W = self.Wresidual_block41(x4W)
        x4W = self.Wresidual_block42(x4W)

        ######################################
        #Amir
        # A = self.decoder_A(x, x1, x2, x4AT,depth, opt)
        A = self.decoder_A(x, x1, x2,g_light, x4AT, depth1, opt)
        T = self.decoder_T(x, x1, x2,g_light, x4AT,depth1, opt)
        T=self.sigT(T)
        # T = self.sigT(self.convT(self.ResT(self.convT1(T))))
        # T = T.repeat(1, 3, 1, 1)
        # J_AT = self.JGen(A=A, t=T, I=x)
        J_AT=torch.mul(A,T)
        J_direct = self.decoder_J(x, x1, x2,g_light, x4J,depth1, opt)

        w = self.decoder_w(x, x1, x2,g_light, x4W,depth1, opt)
        w = self.sigw(self.convw(self.Resw(self.convw1(w))))
        w = w.repeat(1, 3, 1, 1)

        J_total = w*J_direct + (1-w)*J_AT

        return  J_total,J_direct,J_AT,A,T,w,g_light

