import abc
import copy
from torch import nn
from adept.network import SubModule3D
from adept.scripts.local import parse_args


class AdeptMarioNet(SubModule3D):
    #prompted when training starts
    args = {"input_dim": (4, 84, 84), "output_dim": 2}

    def __init__(self, input_dim, output_dim):
        super(AdeptMarioNet, self).__init__()
        #set properties here
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online_conv1 = nn.Conv2d(in_channels=c, out_channels=32,
                                      kernel_size=8, stride=4)
        self.online_conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                                      kernel_size=4, stride=2)
        self.online_conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                                      kernel_size=3, stride=1)

        self.target_conv1 = nn.Conv2d(in_channels=c, out_channels=32,
                                      kernel_size=8, stride=4)
        self.target_conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                                      kernel_size=4, stride=2)
        self.target_conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                                      kernel_size=3, stride=1)


        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    @classmethod
    def from_args(cls, args):

        pass

    def calc_output_dim(dim_size, kernel_size, stride, padding, dilation):
        numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return numerator // stride + 1
    @property
    def _output_shape(self):
        output_dim = 84
        output_dim = self.calc_output_dim(output_dim, 8, 4, 0, 1)
        output_dim = self.calc_output_dim(output_dim, 4, 2, 0, 1)
        output_dim = self.calc_output_dim(output_dim, 3, 1, 0, 1)
        return output_dim
    # def _forward(self, input, internals, **kwargs):
    #     if model == "online":
    #         return self.online(input), {}
    #     elif model == "target":
    #         return self.target(input), {}

    def _new_internals(self):
        return {}