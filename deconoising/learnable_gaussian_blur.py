import torch
import torch.nn as nn
import torch.nn.functional as F


# Taken from https://github.com/pytorch/xla/issues/1289
class GaussianLayer(nn.Module):
    """
    A gaussian layer whose trainable parameter is the standard deviation and we suggest put this
    """

    def __init__(self,
                 input_channels: int,
                 kernel_size: int,
                 pad_type: str = 'zero',
                 stride: list = [1, 1],
                 std: float = None,
                 fixed_std: bool = False):
        """
        Initializer
        :param input_channels: the number of input channels
        :param kernel_size: the size of the gaussian filter
        :param pad_type: the type of the padding
        :param fixed_std: the feature of the paper
        """
        super(GaussianLayer, self).__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride

        #         self.register_buffer("xy_squared_sum", self._generate_grid(kernel_size, input_channels))
        self.xy_squared_sum = nn.Parameter(self._generate_grid(kernel_size, input_channels), requires_grad=False)
        #         self.register_parameter("std", nn.Parameter(torch.tensor(1.)))
        self.std = nn.Parameter(torch.tensor(std or 1).type(torch.float), requires_grad=not fixed_std)


#         self.pad = self._get_pad_layer(pad_type)(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Each time a forward method is called, the model generates a new Gaussian filter
        based on current standard deviation
        :param x: the input
        :return: the de-noised output
        """
        return F.conv2d(F.pad(
            x, [self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2], 'reflect'),
                        self._generate_gaussian(),
                        stride=self.stride,
                        groups=self.input_channels)

    @staticmethod
    def _generate_grid(kernel_size: int, channels: int) -> torch.Tensor:
        """
        This function creates the coordinates of (x, y) for generating Gaussian filters
        :param kernel_size: the size of the square kernel
        :return: the grid to facilitate the calculation
        """
        xv, yv = torch.meshgrid([
            torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
            torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        ])
        xv, yv = xv.type(torch.float).repeat(channels, 1, 1), yv.type(torch.float).repeat(channels, 1, 1)
        # for computation
        return (xv**2 + yv**2) / 2

    def _generate_gaussian(self):
        """
        This function generates the Gaussian filter based on the standard deviation
        :return: the new Gaussian filter
        """
        kernel = torch.exp(-self.xy_squared_sum / self.std**2)

        #         kernel = torch.einsum("ijk, i->ijk", [kernel, (torch.tensor(1.) / kernel.sum(-1).sum(-1))]) This does not work with xla
        kernel = kernel / kernel.sum() * self.input_channels
        return kernel.view(self.input_channels, 1, self.kernel_size, self.kernel_size)
