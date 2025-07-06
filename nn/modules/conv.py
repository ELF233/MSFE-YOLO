# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "NLTEM",
    "GET_TEM",
    "MAMBA",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/ghostnet
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (tuple): Tuple containing:
                - Equivalent kernel (torch.Tensor)
                - Equivalent bias (torch.Tensor)
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            (tuple): Tuple containing:
                - Fused kernel (torch.Tensor)
                - Fused bias (torch.Tensor)
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]








import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GCNBlock(nn.Module):
    """轻量图卷积模块"""

    def __init__(self, in_ch, num_nodes=9):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((num_nodes, num_nodes)),
            nn.Conv2d(in_ch, in_ch // 4, 1),
            nn.BatchNorm2d(in_ch // 4),
            nn.SiLU()
        )
        self.gcn = nn.Sequential(
            nn.Conv1d(num_nodes ** 2, num_nodes ** 2, 1, groups=num_nodes ** 2),  # ([1, 81, 128])
            nn.Conv1d(num_nodes ** 2, in_ch // 4, 1),  # ([1, 128, 128])
            nn.Conv1d(in_ch // 4, in_ch // 2, 1),  # ([1, 128, 81])

        )
        self.rebuild = nn.Conv2d(in_ch // 4, in_ch, 1)

    def auto_hw(self, n):
        """自动找一组最接近正方形的 h×w=n"""
        for i in range(int(math.sqrt(n)), 0, -1):
            if n % i == 0:
                return i, n // i
        return int(math.sqrt(n)), int(math.sqrt(n))

    def forward(self, x):

        B, C, H, W = x.shape  # torch.Size([2, 512, 40, 40])
        nodes = self.proj(x)
        nodes = nodes.reshape(B, -1, C // 4)
        nodes = self.gcn(nodes)

        b, l, c = nodes.shape
        if C // 4 * H * W == l * c:
            nodes = nodes.reshape(B, C // 4, H, W)  # torch.Size([1, 128, 16, 16])
        else:
            h, w = self.auto_hw(l * c // (C // 4))  # 尽可能的保留空间语义
            nodes = nodes.reshape(B, C // 4, h, w)
            nodes = F.interpolate(nodes, size=(H, W), mode="bilinear", align_corners=False)

        return x + self.rebuild(nodes)


class NLTEM(nn.Module):
    def __init__(self, m0, m1, m2, c1, c2):  # c1=
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        self.m0 = m0
        self.m1 = m1
        self.m2 = m2

        self.m_sum = m0 + m1 + m2

        self.Wq = nn.Linear(self.m_sum, self.c1, bias=False)

        self.Wk1 = nn.Linear(self.m0, self.c1, bias=False)
        self.Wv1 = nn.Linear(self.m0, self.c1, bias=False)

        self.Wk2 = nn.Linear(self.m1, self.c1, bias=False)
        self.Wv2 = nn.Linear(self.m1, self.c1, bias=False)

        self.Wk3 = nn.Linear(self.m2, self.c1, bias=False)
        self.Wv3 = nn.Linear(self.m2, self.c1, bias=False)

        self.conv1 = nn.Conv1d(in_channels=self.c1, out_channels=self.c1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.c1, out_channels=self.c1, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=self.c1, out_channels=self.c1, kernel_size=1)

        self.act = nn.Sigmoid()

        self.gfm = GCNBlock(in_ch=self.c1)

        self.conv2d_1 = nn.Conv2d(self.c1, self.m0, 1)
        self.conv2d_2 = nn.Conv2d(self.c1, self.m1, 1)
        self.conv2d_3 = nn.Conv2d(self.c1, self.m2, 1)

    def liner_channl(self, in_x, ch):
        # print(in_x.shape)
        b, c, h, w = in_x.shape
        x_perm = in_x.permute(0, 2, 3, 1)
        if ch == self.m_sum:
            x_flat = x_perm.reshape(-1, self.m_sum)
            x_out = self.Wq(x_flat)
            x_out = x_out.reshape(b, h, w, self.c1)
            x_out = x_out.permute(0, 3, 1, 2)
            return x_out

        elif ch == self.m0:
            x_flat = x_perm.reshape(-1, self.m0)
            x_k = self.Wk1(x_flat)
            x_k = x_k.reshape(b, h, w, self.c1)

            x_v = self.Wv1(x_flat)
            x_v = x_v.reshape(b, h, w, self.c1)

        elif ch == self.m1:
            x_flat = x_perm.reshape(-1, self.m1)
            x_k = self.Wk2(x_flat)
            x_k = x_k.reshape(b, h, w, self.c1)

            x_v = self.Wv2(x_flat)
            x_v = x_v.reshape(b, h, w, self.c1)

        else:
            x_flat = x_perm.reshape(-1, self.m2)
            x_k = self.Wk3(x_flat)
            x_k = x_k.reshape(b, h, w, self.c1)

            x_v = self.Wv3(x_flat)
            x_v = x_v.reshape(b, h, w, self.c1)

        x_k = x_k.permute(0, 3, 1, 2)
        x_v = x_v.permute(0, 3, 1, 2)
        return x_k, x_v

    def repeat_tt(self, x, q, ch, old_x):
        Tk, Tv = self.liner_channl(x, ch)  # b1, self.c1, h1, w1

        Tkq = q * Tk
        tb, tc, th, tw = Tkq.shape
        Tkq = Tkq.view(tb, tc, th * tw)  # b,c, h*w

        if ch == self.m0:
            Tq = self.conv1(Tkq)
        elif ch == self.m1:
            Tq = self.conv2(Tkq)
        else:
            Tq = self.conv3(Tkq)  # b,c, h*w

        Tq = Tq.view(tb, tc, th, tw)

        Ta = self.act(Tq * Tk)  # b1, self.c1, h1, w1
        Tv_Ta = Tv * Ta

        Tg = self.gfm(Tv_Ta)
        b, c, h, w = old_x.shape
        Tg_Ta = F.interpolate(Ta * Tg, size=(h, w), mode="bilinear", align_corners=False)  # 返回原尺寸

        if ch == self.m0:
            To = self.conv2d_1(Tg_Ta) * old_x
        elif ch == self.m1:
            To = self.conv2d_2(Tg_Ta) * old_x
        else:
            To = self.conv2d_3(Tg_Ta) * old_x

        return To

    def forward(self, x):
        old_x1, old_x2, old_x3 = x[0], x[1], x[2]
        b1, c1, h1, w1 = old_x1.shape
        b2, c2, h2, w2 = old_x2.shape
        b3, c3, h3, w3 = old_x3.shape

        x1 = F.interpolate(old_x1, size=(h2, w2), mode="bilinear", align_corners=False)
        x2 = F.interpolate(old_x2, size=(h2, w2), mode="bilinear", align_corners=False)
        x3 = F.interpolate(old_x3, size=(h2, w2), mode="bilinear", align_corners=False)

        concat_x = torch.cat([x1, x2, x3], 1)
        Tq = self.liner_channl(concat_x, self.m_sum)

        T0 = self.repeat_tt(x1, Tq, self.m0, old_x1)
        T1 = self.repeat_tt(x2, Tq, self.m1, old_x2)
        T2 = self.repeat_tt(x3, Tq, self.m2, old_x3)

        # print('输入:', old_x1.shape, old_x2.shape, old_x3.shape)
        # print('输出:', T0.shape, T1.shape, T2.shape)
        return [T0, T1, T2]  # 返回列表


class GET_TEM(nn.Module):

    def __init__(self, c1, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        """
        2   torch.Size([1, 256, 8, 8])
        1   torch.Size([1, 128, 16, 16])
        0   torch.Size([1, 64, 32, 32])
        """
        out = x[self.idx]
        # print(self.idx, out.shape)
        return out








from mmcv.runner import BaseModule
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

from .rs_mamba_cd import VSSBlock
import torch.nn.functional as F

class MAMBA(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 解析参数：args[0]是ch，args[1:]是YAML中的参数
        if len(args) == 8:  # ch + 7个参数
            ch, emb_size, num_heads, depth, expansion_rate, dropout_rate, um, tp = args
        else:
            raise ValueError(f"MAMBA expects 8 arguments (ch + 7 configs), got {len(args)}")

        self.tp = tp
        self.um = um
        self.relu = nn.ReLU(inplace=True)
        out_channel = emb_size // 4
        # 其余代码不变...

        self.branch0 = nn.Sequential(
            BasicConv2d(emb_size, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(emb_size, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(emb_size, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=5, padding=2),
        )
        self.reasoning0 = VSSBlock(out_channel, drop_path=dropout_rate)
        self.reasoning1 = VSSBlock(out_channel, drop_path=dropout_rate)
        self.reasoning2 = VSSBlock(out_channel, drop_path=dropout_rate)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x0 = x0.permute(0, 3, 2, 1)
        x1 = x1.permute(0, 3, 2, 1)
        x2 = x2.permute(0, 3, 2, 1)

        x0 = self.reasoning0(x0)

        x1 = self.reasoning1(x1)
        x2 = self.reasoning2(x2)
        x_cat = torch.cat((x0, x1, x2), 1)

        x_cat = x_cat.permute(0, 3, 2, 1)

        # print(x_cat.shape, x.shape)  # torch.Size([16, 256, 17, 63]) torch.Size([16, 1024, 17, 21])

        x_cat = F.interpolate(x_cat, size=(x.shape[2], x.shape[3]*(x.shape[1] // x_cat.shape[1])), mode='bilinear', align_corners=False)

        # print(x_cat.shape, x.shape)

        x_cat = x_cat.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])  # torch.Size([1, 1024, 8, 6])

        # print(x_cat.shape, x.shape)

        x = self.relu(x_cat + x)

        return x

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x):
        # Assumes x is a list with two tensors to multiply
        assert len(x) == 2, "Multiply layer requires exactly two input tensors"
        print("Shape of input tensors:", x[0].shape, x[1].shape)
        return x[0] * x[1]






