from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools

from functools import partial
from model.util.mesh_util import *


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal., kinda like std

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    # define the initialization rules for different layers
    def init_func(m):

        # name of one layer
        classname = m.__class__.__name__

        # init. regular conv, fc layers
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            # init the weights
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            # init the bias (if have it)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # init. batchNorm layers
        elif classname.find('BatchNorm2d') != -1:

            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    # multi-GPU settings, default: single-GPU training/test
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    
    # return is not necessary, not used anywhere
    return net


def imageSpaceRotation(xy, rot):
    '''
    args:
        xy: (B, 2, N) input
        rot: (B, 2) x,y axis rotation angles

    rotation center will be always image center (other rotation center can be represented by additional z translation)
    '''
    disp = rot.unsqueeze(2).sin().expand_as(xy)
    return (disp * xy).sum(dim=1)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv2dSame(nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        # self.weight = self.net[1].weight
        # self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)

class rgb_rendering_unet(nn.Module):

    def __init__(self, c_len_in, c_len_out, opt=None):

        super(rgb_rendering_unet, self).__init__()

        # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B, 8,192,128) -----> skip-0: (B, 8,192,128)
        # (B,    8,192,128) |   conv2d(    i8,o16,k4,s2,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64) -----> skip-1: (B,16, 96, 64)
        # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),     LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64)
        # (B,   16, 96, 64) |   conv2d(   i16,o32,k4,s2, b),     LeakyReLU(0.2), Dp(0.1) | (B,32, 48, 32)
        c_len_1 = c_len_in
        self.rendering_enc_conv2d_1 = nn.Sequential(Conv2dSame(c_len_in,c_len_1,kernel_size=3,bias=False), nn.BatchNorm2d(c_len_1,affine=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))
        c_len_2 = c_len_1 * 2
        self.rendering_enc_conv2d_2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c_len_1,c_len_2,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm2d(c_len_2,affine=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))
        c_len_3 = c_len_2
        self.rendering_enc_conv2d_3 = nn.Sequential(Conv2dSame(c_len_2,c_len_3,kernel_size=3,bias=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))
        c_len_4 = c_len_3 * 2
        self.rendering_enc_conv2d_4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c_len_3,c_len_4,kernel_size=4,padding=0,stride=2,bias=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))

        # (B,   32, 48, 32) | deconv2d(   i32,o16,k4,s2, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        # (B,16+16, 96, 64) | deconv2d(i16+16, o8,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128) <----- skip-1: (B,16, 96, 64)
        # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128)
        # (B,  8+8,192,128) | deconv2d(  i8+8, o3,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 3,384,256) <----- skip-0: (B, 8,192,128)
        # (B,    3,384,256) |   conv2d(    i3, o3,k3,s1, b), Tanh                        | (B, 3,384,256)
        self.rendering_dec_conv2d_1 = nn.Sequential(nn.ConvTranspose2d(c_len_4,c_len_3,kernel_size=4,stride=2,padding=1,bias=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_2 = nn.Sequential(Conv2dSame(c_len_3,c_len_2,kernel_size=3,bias=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_3 = nn.Sequential(nn.ConvTranspose2d(c_len_2*2,c_len_1,kernel_size=4,stride=2,padding=1,bias=False), nn.BatchNorm2d(c_len_1,affine=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_4 = nn.Sequential(Conv2dSame(c_len_1,c_len_1,kernel_size=3,bias=False), nn.BatchNorm2d(c_len_1,affine=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_5 = nn.Sequential(nn.ConvTranspose2d(c_len_1*2,c_len_out,kernel_size=4,stride=2,padding=1,bias=False), nn.BatchNorm2d(c_len_out,affine=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_6 = nn.Sequential(Conv2dSame(c_len_out,c_len_out,kernel_size=3,bias=True), nn.Tanh())

    def forward(self, x):

        # init.
        skip_list = []

        # encoder
        x = self.rendering_enc_conv2d_1(x); skip_list.append(x)             # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B, 8,192,128) -----> skip-0: (B, 8,192,128)
        x = self.rendering_enc_conv2d_2(x); skip_list.append(x)             # (B,    8,192,128) |   conv2d(    i8,o16,k4,s2,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64) -----> skip-1: (B,16, 96, 64)
        x = self.rendering_enc_conv2d_3(x)                                  # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),     LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64)
        x = self.rendering_enc_conv2d_4(x); skip_list.append(x)             # (B,   16, 96, 64) |   conv2d(   i16,o32,k4,s2, b),     LeakyReLU(0.2), Dp(0.1) | (B,32, 48, 32)

        # decoder
        x = self.rendering_dec_conv2d_1(x)                                  # (B,   32, 48, 32) | deconv2d(   i32,o16,k4,s2, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        x = self.rendering_dec_conv2d_2(x)                                  # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        x = self.rendering_dec_conv2d_3(torch.cat([skip_list[1],x], dim=1)) # (B,16+16, 96, 64) | deconv2d(i16+16, o8,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128) <----- skip-1: (B,16, 96, 64)
        x = self.rendering_dec_conv2d_4(x)                                  # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128)
        x = self.rendering_dec_conv2d_5(torch.cat([skip_list[0],x], dim=1)) # (B,  8+8,192,128) | deconv2d(  i8+8, o3,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 3,384,256) <----- skip-0: (B, 8,192,128)
        x = self.rendering_dec_conv2d_6(x)                                  # (B,    3,384,256) |   conv2d(    i3, o3,k3,s1, b), Tanh                        | (B, 3,384,256)

        return x

class Conv3dSame(nn.Module):
    '''3D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReplicationPad3d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb, ka, kb)),
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

    def forward(self, x):

        return self.net(x)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.eps = 1e-9

    def __call__(self, input, target_is_real):
        if target_is_real:
            # if True, then the loss encourages input to increase
            return -1.*torch.mean(torch.log(input + self.eps))
        else:
            # if False, then the loss encourages input to decrease
            return -1.*torch.mean(torch.log(1 - input + self.eps))

class Unet3D(nn.Module):

    def __init__(self, c_len_in, c_len_out, opt=None):

        super(Unet3D, self).__init__()

        # (BV,8,32,48,32) | conv3d(k3,s1,i8,o8,nb), BN3d, LearkyReLU(0.2) | (BV,8,32,48,32) ------> skip-0: (BV,8,32,48,32)
        c_len_1 = 8
        self.conv3d_pre_process = nn.Sequential(Conv3dSame(c_len_in,c_len_1,kernel_size=3,bias=False), nn.BatchNorm3d(c_len_1,affine=True), nn.LeakyReLU(0.2,True))

        # (BV,8,32,48,32) | conv3d(k4,s2,i8,o16,nb), BN3d, LeakyReLU(0.2) | (BV,16,16,24,16) ------> skip-1: (BV,16,16,24,16)
        c_len_2 = 16
        self.conv3d_enc_1 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c_len_1,c_len_2,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_2,affine=True), nn.LeakyReLU(0.2,True))

        # (BV,16,16,24,16) | conv3d(k4,s2,i16,o32,b), LeakyReLU(0.2) | (BV,32,8,12,8)
        c_len_3 = 32
        self.conv3d_enc_2 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c_len_2,c_len_3,kernel_size=4,padding=0,stride=2,bias=True), nn.LeakyReLU(0.2,True))

        # (BV,32,8,12,8) | DeConv3d(k4,s2,i32,o16,b), ReLU | (BV,16,16,24,16)
        self.deconv3d_dec_2 = nn.Sequential(nn.ConvTranspose3d(c_len_3,c_len_2,kernel_size=4,stride=2,padding=1,bias=True), nn.ReLU(True))

        # (BV,16+16,16,24,16) | DeConv3d(k4,s2,i32,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-1: (BV,16,16,24,16)
        self.deconv3d_dec_1 = nn.Sequential(nn.ConvTranspose3d(c_len_2*2,c_len_1,kernel_size=4,stride=2,padding=1,bias=False), nn.BatchNorm3d(c_len_1,affine=True), nn.ReLU(True))

        # (BV,8+8,32,48,32) | Conv3d(k3,s1,i16,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-0: (BV,8,32,48,32)
        self.conv3d_final_process = nn.Sequential(Conv3dSame(c_len_1*2,c_len_out,kernel_size=3,bias=False), nn.BatchNorm3d(c_len_out,affine=True), nn.ReLU(True))

    def forward(self, x):
        """
        e.g. in-(BV,8,32,48,32), out-(BV,8,32,48,32)
        """

        skip_encoder_list = []

        # (BV,8,32,48,32) | conv3d(k3,s1,i8,o8,nb), BN3d, LearkyReLU(0.2) | (BV,8,32,48,32) ------> skip-0: (BV,8,32,48,32)
        x = self.conv3d_pre_process(x)
        skip_encoder_list.append(x)

        # (BV,8,32,48,32) | conv3d(k4,s2,i8,o16,nb), BN3d, LeakyReLU(0.2) | (BV,16,16,24,16) ------> skip-1: (BV,16,16,24,16)
        x = self.conv3d_enc_1(x)
        skip_encoder_list.append(x)

        # (BV,16,16,24,16) | conv3d(k4,s2,i16,o32,b), LeakyReLU(0.2) | (BV,32,8,12,8)
        x = self.conv3d_enc_2(x)

        # (BV,32,8,12,8) | DeConv3d(k4,s2,i32,o16,b), ReLU | (BV,16,16,24,16)
        x = self.deconv3d_dec_2(x)

        # (BV,16+16,16,24,16) | DeConv3d(k4,s2,i32,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-1: (BV,16,16,24,16)
        x = torch.cat([skip_encoder_list[1], x], dim=1)
        x = self.deconv3d_dec_1(x)

        # (BV,8+8,32,48,32) | Conv3d(k3,s1,i16,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-0: (BV,8,32,48,32)
        x = torch.cat([skip_encoder_list[0], x], dim=1)
        x = self.conv3d_final_process(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3
  

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=1)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=1)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def get_inplanes():
    return [8, 16, 32, 64]
def get_inplanes_TMM():
    return [8, 16, 16, 8]

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes_TMM(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model




class VolumeHourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch',upsample_mode="bicubic"):
        super(VolumeHourGlass, self).__init__()
        self.num_modules = num_modules # default: 1
        self.depth = depth # default: 2
        self.features = num_features # default: 256
        self.norm = norm # default: group
        self.upsample_mode = upsample_mode # default: bicubic

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), BasicBlock(self.features, self.features))

        self.add_module('b2_' + str(level), BasicBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), BasicBlock(self.features, self.features))

        self.add_module('b3_' + str(level), BasicBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool3d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1: # default: 2
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        if self.upsample_mode == "bicubic":
            up2 = F.interpolate(low3, scale_factor=2, mode='trilinear', align_corners=True)
        elif self.upsample_mode == "nearest":
            up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        else:
            print("Error: undefined self.upsample_mode {}!".format(self.upsample_mode))

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)











