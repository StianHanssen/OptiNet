import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn import functional as F
from utils import init_weights

_CONV3D = None

def conv3x3x3(in_channels, out_channels, stride=1, bias=True):
    # 3x3x3 convolution with padding
    return _CONV3D(in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=stride,
                   padding=1,
                   bias=bias)

class Conv2_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=True):
        super().__init__()
        kernel_size = self.tuplify(kernel_size)
        stride = self.tuplify(stride)
        padding = self.tuplify(padding)
        dilation = self.tuplify(dilation)
        depth = kernel_size[0]
        slice_dim = (kernel_size[1] + kernel_size[2])//2
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculation from the paper to make number of parameters equal to 3D convolution
        self.hidden_size = int((depth * slice_dim**2 * in_channels * out_channels) / 
                               (slice_dim**2 * in_channels + depth * out_channels))

        self.conv2d = nn.Conv2d(in_channels, self.hidden_size, kernel_size[1:], stride[1:], padding[1:], bias=bias)
        self.conv1d = nn.Conv1d(self.hidden_size, out_channels, kernel_size[0], stride[0], padding[0], bias=bias)

    def forward(self, x):
        #2D convolution
        b, c, d, h, w = x.size()
        # Rearrange depth and channels and combine depth with batch size
        # This way each slice becomes an individual case to do 2D convolution on
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*d, c, h, w)
        x = F.relu(self.conv2d(x))
        
        #1D convolution
        c, h, w = x.size()[1:]
        # Prepare for new rearrangement by parting depth and batch size
        x = x.view(b, d, c, h, w)
        # Rearrange shape to (batch size, height, width, channels, depth)
        # Combine batch size, height and width
        # This way each line of pixels depth-wise becomes an individual case
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*h*w, c, d)
        x = self.conv1d(x)

        #Final output
        final_c, final_d = x.size()[1:]
        # Split batch, heigh and width again 
        x = x.view(b, h, w, final_c, final_d)
        # Rearrange dimensions back to the original order
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x
    
    @staticmethod
    def tuplify(arg):
        # Turns a single int i to a tuple (i, i, i)
        # If input is tuple/list it will assert length of 3 and return input
        if isinstance(arg, int):
            return (arg,) * 3
        assert isinstance(arg, tuple) or isinstance(arg, list), "Expected list or tuple, but got " + str(type(arg)) + "."
        assert len(arg) == 3, "Expected " + str(arg) + " to have 3 values!"
        return arg

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None, stride=1, bias=True):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3x3(in_channels, out_channels, stride, bias)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels, stride, bias)
        self.bn2 = nn.BatchNorm3d(out_channels)

        net_list = [self.conv1,
                    self.bn1,
                    self.relu,
                    self.conv2,
                    self.bn2,
                    self.relu]
        
        if down_sample is not None:
            net_list.append(down_sample)
        
        self.net = nn.Sequential(*net_list)
    
    def forward(self, x):
        return self.net(x)

class AMDModel(nn.Module):
    def __init__(self, conv3d_module, stride=1, bias=True):
        super(AMDModel, self).__init__()

        global _CONV3D
        _CONV3D = conv3d_module
        
        self.down_sample1 = nn.MaxPool3d((2, 4, 4), (2, 4, 4))
        self.down_sample2 = nn.MaxPool3d(2, 2)
                                                                                          # (1, 32, 256, 256)
        self.feature = nn.Sequential(Block(1, 64, self.down_sample1, stride, bias),       # (64, 16, 64, 64)
                                     Block(64, 128, self.down_sample1, stride, bias),     # (128, 8, 16, 16)
                                     Block(128, 256, self.down_sample1, stride, bias),    # (256, 4, 4, 4)
                                     Block(256, 512, self.down_sample2, stride, bias),    # (512, 2, 2, 2)
                                     conv3x3x3(512, 512, stride, bias),
                                     nn.ReLU(inplace=True),
                                     self.down_sample2                      # (512, 1, 1, 1) = (512)
                                     )

        self.classifier = nn.Sequential(nn.Linear(512, 256),       # (256)
                                        nn.ReLU(),
                                        nn.Linear(256, 1),          # (1)
                                        nn.Sigmoid())

        for m in self.modules():
            init_weights(m)

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
