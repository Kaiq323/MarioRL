from torch import nn
from torch.nn import functional as F

from adept.network import SubModule3D


class AdeptMarioNet(SubModule3D):
    # prompted when training starts
    args = {}

    def __init__(self, in_shape, id):
        super(AdeptMarioNet, self).__init__(in_shape, id)
        # set properties here
        c, h, w = in_shape

        self._in_shape = in_shape
        self._out_shape = None

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, args, in_shape, id):
        return cls(in_shape, id)

    @property
    def _output_shape(self):
        if self._out_shape is None:
            output_dim = calc_output_dim(self._in_shape[1], 8, 4, 0, 1)
            output_dim = calc_output_dim(output_dim, 4, 2, 0, 1)
            output_dim = calc_output_dim(output_dim, 3, 1, 0, 1)
            self._out_shape = 64, output_dim, output_dim
        return self._out_shape

    def _forward(self, xs, internals, **kwargs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        return xs, {}

    def _new_internals(self):
        return {}


def calc_output_dim(dim_size, kernel_size, stride, padding, dilation):
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1
