import torch
import torch.nn as nn
import numpy as np


class AttentionGate(nn.Module):
    def __init__(self, x_ch, g_ch, mid_ch, scale):
        super(AttentionGate, self).__init__()
        self.scale = scale
        self.conv_g = nn.Conv2d(in_channels=g_ch, out_channels=mid_ch, kernel_size=1, bias=True)
        self.conv_x = nn.Conv2d(in_channels=x_ch, out_channels=mid_ch, kernel_size=1, stride=scale)
        self.relu = nn.ReLU(inplace=True)
        self.conv_epsi = nn.Conv2d(in_channels=mid_ch, out_channels=1, kernel_size=1, bias=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x, g):
        attn = self.relu(self.conv_g(g) + self.conv_x(x))
        attn = self.sigm(self.conv_epsi(attn))
        return nn.functional.interpolate(attn, scale_factor=self.scale) * x


class JGenerate(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(JGenerate, self).__init__()
        self.epsilon = epsilon

    def forward(self, A, t, I):
        # self.epsilon = 1e-10
        return (I - A)/(t + self.epsilon) + A


class BReLU(nn.Module):
    def __init__(self, inplace=True):
        super(BReLU, self).__init__()
        # self.relu = nn.ReLU(inplace=inplace)
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # out = self.relu(x)
        # return 1 - self.relu(1 - out)
        return self.relu6(6*x)/6


class DarkChannel(nn.Module):
    def __init__(self, window_size):
        super(DarkChannel, self).__init__()
        self.mxpool = nn.MaxPool2d(kernel_size=window_size, stride=1, padding=(window_size-1)//2)

    def forward(self, x):
        neg_x_pooled = self.mxpool(-1*x)
        return -1*(neg_x_pooled.max(dim=1, keepdim=True)[0])


def generate_mask(J, A, t, t_th, A_th, t_sl, A_sl, eps=1e-4):
    # closer to one at unclear regions
    t_relu = torch.nn.functional.sigmoid(t_sl*(t_th - t))
    A_relu = torch.nn.functional.sigmoid(A_sl*((A / (J + eps)).mean(dim=1, keepdim=True) - A_th))
    total_relu = t_relu * A_relu
    return total_relu


def generate_J(I, A, t, eps=1e-4):
    return (I - A)/(t + eps) + A


def correlation_coef(x, y):
    vx = x - torch.mean(x, dim=(1, 2, 3), keepdim=True)
    vy = y - torch.mean(y, dim=(1, 2, 3), keepdim=True)
    return (torch.sum(vx * vy, dim=(1, 2, 3)) / (torch.sqrt(torch.sum(vx ** 2, dim=(1, 2, 3))) * torch.sqrt(torch.sum(vy ** 2, dim=(1, 2, 3))))).mean()


class EdgePreservingMSE(nn.Module):
    def __init__(self, factor=0.1, std_dev=1.0):
        super(EdgePreservingMSE, self).__init__()
        self.mu = factor

        eye = torch.eye(3).type(torch.DoubleTensor).unsqueeze(-1).unsqueeze(-1)
        filter_size = 5
        generated_filters = torch.from_numpy(np.array([[np.exp(-(ix*ix + iy*iy)/(2*std_dev*std_dev))/np.sqrt(2*np.pi*std_dev*std_dev)
                                                       for ix in range(-filter_size//2 + 1, filter_size//2 + 1)]
                                                       for iy in range(-filter_size//2 + 1, filter_size//2 + 1)], dtype=np.float64))
        self.gaussian_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=filter_size, padding=(filter_size - 1)//2, bias=False)
        self.gaussian_filter.weight.data.copy_(generated_filters*eye)

        sobel_filter = torch.from_numpy(np.array([[1, 0, -1],
                                                  [2, 0, -2],
                                                  [1, 0, -1]], dtype=np.float64).reshape([1, 1, 3, 3])) * eye
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.sobel_filter_horizontal.weight.data.copy_(sobel_filter)

        self.sobel_filter_vertical = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.sobel_filter_vertical.weight.data.copy_(sobel_filter.transpose(2, 3))

        self.sobel_filter_vertical.weight.requires_grad = False
        self.sobel_filter_horizontal.weight.requires_grad = False
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, net_output, target):
        with torch.no_grad():
            blurred_img = self.gaussian_filter(target)
            grad_x = self.sobel_filter_horizontal(blurred_img)
            grad_y = self.sobel_filter_vertical(blurred_img)
            factor = 1 + self.mu*torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return (factor*((net_output - target)**2)).mean()


class PReLU2sided(nn.Module):
    def __init__(self, init_negative=0.1, init_positive=0.1):
        super(PReLU2sided, self).__init__()
        # self.relu = nn.ReLU(inplace=inplace)
        self.relu = nn.PReLU(init=init_negative)
        self.reversed_relu = nn.PReLU(init=init_positive)

    def forward(self, x):
        out = self.relu(x)
        return 1 - self.reversed_relu(1 - out)


class LeakyReLU2sided(nn.Module):
    def __init__(self, init_negative=0.01, init_positive=0.01, inplace=True):
        super(LeakyReLU2sided, self).__init__()
        # self.relu = nn.ReLU(inplace=inplace)
        self.relu = nn.LeakyReLU(negative_slope=init_negative, inplace=inplace)
        self.reversed_relu = nn.LeakyReLU(negative_slope=init_positive, inplace=inplace)

    def forward(self, x):
        out = self.relu(x)
        return 1 - self.reversed_relu(1 - out)


class func_clamp_noderiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, min_val, max_val):
        ctx._mask = (i.ge(min_val) * i.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        mask = torch.autograd.Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None, None
        # return grad_output, None, None
def cvtLab(RGB):
    #input is bgr actually
	# threshold definition
    N=RGB.shape[0]
    RGB=RGB.permute(0,2,3,1)
    RGB=torch.flatten(RGB,0,2)
    T = 0.008856

    # matrix for converting RGB to LUV color space
    cvt_XYZ = torch.tensor([[0.180423,0.35758,0.412453],[0.072169,0.71516,0.212671],[0.950227,0.119193,0.019334]]).cuda()

    # convert RGB to XYZ
    XYZ = torch.matmul(RGB,torch.transpose(cvt_XYZ,0,1))

    # normalise for D65 white point
    XYZ /= torch.tensor([[1.088754, 1., 0.950456]]).cuda()*100

    mask = (torch.gt(XYZ,T)).float()

    fXYZ = XYZ**(1/3)*mask + (1.-mask)*(7.787*XYZ + 0.137931)

    M_cvtLab = torch.tensor([ [0., 116., 0.],  [0, -500., 500.],[-200., 200., 0]]).cuda()
    Lab = torch.matmul(fXYZ, torch.transpose(M_cvtLab,0,1)) + torch.tensor([[  0.,0,-16]]).cuda()
    mask = (torch.eq(Lab,0.)).float()

    Lab += mask * 1e-4
    Lab=Lab.reshape(N,256,256,3)
    Lab=Lab.permute(0,3,1,2)
    return Lab


def smoothness(inputs,albedos):
    # albedos=albedos.permute(0,2,3,1)
    Gx = 1 / 2 * (torch.tensor([[-1, 1]], dtype=torch.float32).unsqueeze(0)).unsqueeze(0)
    Gy = 1 / 2 * ((torch.tensor([[-1], [1]], dtype=torch.float32)).unsqueeze(0)).unsqueeze(0)
    Gx_3 = torch.cat([Gx,Gx,Gx] ,1).cuda()
    Gy_3 = torch.cat([Gy,Gy,Gy], 1).cuda()
    # albedo_lab = torch.reshape(cvtLab(torch.reshape(albedos,[-1,3])),[-1,256,256,3])

    aGx = torch.nn.functional.conv2d(torch.nn.functional.pad(albedos,pad=(0,1,0,0)), Gx_3,stride=(1, 1))
    aGy = torch.nn.functional.conv2d(torch.nn.functional.pad(albedos,pad=(0,0,1,0)), Gy_3,stride=(1, 1))
    aGxy = torch.cat([aGx, aGy], axis=1)

    # compute pixel-wise smoothness weights by angle distance between neighbour pixels' chromaticities
    inputs_pad = torch.nn.functional.pad(inputs, pad=(0, 1, 0, 1))
    chroma_pad = torch.nn.functional.normalize(inputs_pad, dim=1, p=2)

    chroma = chroma_pad[:, :, :-1, :-1]
    chroma_X = chroma_pad[:, :, :-1, 1:]
    chroma_Y = chroma_pad[:, :, 1:, :-1]
    chroma_Gx = torch.sum(chroma * chroma_X, axis=1, keepdim=True) ** 2. - 1.
    chroma_Gy = torch.sum(chroma * chroma_Y, axis=1, keepdim=True) ** 2. - 1.
    chroma_Gx = torch.exp(chroma_Gx / 0.0001)
    chroma_Gy = torch.exp(chroma_Gy / 0.0001)
    chroma_Gxy = torch.cat([chroma_Gx, chroma_Gy], 1)

    int_pad = torch.sum(inputs_pad ** 2., axis=1, keepdim=True)
    int = int_pad[:, :, :-1, :-1]
    int_X = int_pad[:, :, :-1, 1:]
    int_Y = int_pad[:, :, 1:, :-1]

    int_Gx = torch.where(int < int_X, int, int_X)
    int_Gy = torch.where(int < int_Y, int, int_Y)
    int_Gx = 1. + torch.exp(- int_Gx / 8)
    int_Gy = 1. + torch.exp(- int_Gy / .8)
    int_Gxy = torch.cat([int_Gx, int_Gy], axis=1)

    Gxy_weights = int_Gxy * chroma_Gxy
    albedo_smt_error = torch.mean(torch.abs(aGxy) * Gxy_weights)


    return albedo_smt_error

