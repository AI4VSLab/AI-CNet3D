import torch.nn as nn
import torch

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(
                name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError(
            "padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError(
            "out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(
            1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,
                             num_heads=num_heads, channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(
            3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(
            0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))
        attn_skip = attn.reshape(B, H, W, D, C).permute(
            0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class EPABlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            return_attn=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(
            1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,
                             num_heads=num_heads, channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        self.conv8 = nn.Sequential(nn.Dropout3d(
            0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, input_size, hidden_size))
        self.return_attn = return_attn

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        epa = self.epa_block(self.norm(x))
        attn = x + self.gamma * epa

        attn_skip = attn.reshape(B, H, W, D, C).permute(
            0, 4, 1, 2, 3)  # (B, C, H, W, D)

        x = attn_skip + self.conv8(attn_skip)

        if self.return_attn:
            return x, epa.reshape(B, H, W, D, C).permute(
                0, 4, 1, 2, 3)

        return x


class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """

    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(
            B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @
                   k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)
                ).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class HemiretinalNeuronAxonCrossAttentionBlock(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            return_attn=False
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(
            1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.cross_block1 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block2 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block3 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block4 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block5 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block6 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block7 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block8 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.conv8 = nn.Sequential(nn.Dropout3d(
            0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.return_attn = return_attn

    def forward(self, x):

        B, C, H, W, D = x.shape

        x_skip = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        xB, xC, xH, xW, xD = x.shape

        # split features by depth axis
        depth = int(x.shape[2]/2)

        superior = x[:, :, :depth, :, :]
        inferior = x[:, :, depth:, :, :]

        # split features by width axis
        width = int(x.shape[4]/2)

        sup_mac = superior[:, :, :, :, :width]
        sup_nerve = superior[:, :, :, :, width:]

        inf_mac = inferior[:, :, :, :, :width]
        inf_nerve = inferior[:, :, :, :, width:]

        B, C, H, W, D = sup_mac.shape

        sup_mac = sup_mac.reshape(B, C, H * W * D).permute(0, 2, 1)
        sup_mac = self.norm1(sup_mac)
        sup_nerve = sup_nerve.reshape(B, C, H * W * D).permute(0, 2, 1)
        sup_nerve = self.norm2(sup_nerve)

        inf_mac = inf_mac.reshape(B, C, H * W * D).permute(0, 2, 1)
        inf_mac = self.norm1(inf_mac)
        inf_nerve = inf_nerve.reshape(B, C, H * W * D).permute(0, 2, 1)
        inf_nerve = self.norm2(inf_nerve)

        attn_out_1 = self.cross_block1(sup_mac, inf_mac)
        attn_out_2 = self.cross_block2(inf_mac, sup_mac)

        attn_out_3 = self.cross_block3(sup_nerve, inf_nerve)
        attn_out_4 = self.cross_block4(inf_nerve, sup_nerve)

        attn_out_5 = self.cross_block5(sup_nerve, sup_mac)
        attn_out_6 = self.cross_block6(sup_mac, sup_nerve)

        attn_out_7 = self.cross_block7(inf_nerve, inf_mac)
        attn_out_8 = self.cross_block8(inf_mac, inf_nerve)

        mac_attn = torch.cat((attn_out_1, attn_out_2), dim=1)
        nerve_attn = torch.cat((attn_out_3, attn_out_4), dim=1)

        sup_attn = torch.cat((attn_out_5, attn_out_6), dim=1)
        inf_attn = torch.cat((attn_out_7, attn_out_8), dim=1)

        mac_nerve = torch.cat((mac_attn, nerve_attn), dim=1)
        sup_inf = torch.cat((sup_attn, inf_attn), dim=1)

        attn = x_skip + self.gamma * (mac_nerve + sup_inf)

        attn_skip = attn.reshape(xB, xH, xW, xD, xC).permute(
            0, 4, 1, 2, 3)  # (B, C, H, W, D)

        x = attn_skip + self.conv8(attn_skip)

        if self.return_attn:

            mac_attn = mac_attn.reshape(
                xB, xH, xW, xD//2, xC).permute(0, 4, 1, 2, 3)
            nerve_attn = nerve_attn.reshape(
                xB, xH, xW, xD//2, xC).permute(0, 4, 1, 2, 3)
            return x, mac_attn, nerve_attn

        return x


class HemiretinalCrossAttentionBlock(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            return_attn=False
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(
            1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.cross_block1 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block2 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.conv8 = nn.Sequential(nn.Dropout3d(
            0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.return_attn = return_attn

    def forward(self, x):

        B, C, H, W, D = x.shape

        x_skip = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        xB, xC, xH, xW, xD = x.shape

        # split features by depth axis
        depth = int(x.shape[2]/2)

        superior = x[:, :, :depth, :, :]
        inferior = x[:, :, depth:, :, :]

        B, C, H, W, D = superior.shape

        superior = superior.reshape(B, C, H * W * D).permute(0, 2, 1)
        superior = self.norm1(superior)
        inferior = inferior.reshape(B, C, H * W * D).permute(0, 2, 1)
        inferior = self.norm2(inferior)

        attn_out_1 = self.cross_block1(superior, inferior)
        attn_out_2 = self.cross_block2(inferior, superior)

        attn = x_skip + self.gamma * torch.cat((attn_out_1, attn_out_2), dim=1)

        attn_skip = attn.reshape(xB, xH, xW, xD, xC).permute(
            0, 4, 1, 2, 3)  # (B, C, H, W, D)

        x = attn_skip + self.conv8(attn_skip)

        if self.return_attn:

            sup_inf_attn = attn_out_1.reshape(
                xB, xH//2, xW, xD, xC).permute(0, 4, 1, 2, 3)
            inf_sup_attn = attn_out_2.reshape(
                xB, xH//2, xW, xD, xC).permute(0, 4, 1, 2, 3)
            return x, sup_inf_attn, inf_sup_attn

        return x


class NeuronAxonCrossAttentionBlock(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            return_attn=False
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(
            1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.cross_block1 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.cross_block2 = CrossAttention(hidden_size=hidden_size,
                                           num_heads=num_heads, attn_drop_rate=dropout_rate)
        self.conv8 = nn.Sequential(nn.Dropout3d(
            0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.return_attn = return_attn

    def forward(self, x):

        B, C, H, W, D = x.shape

        x_skip = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        xB, xC, xH, xW, xD = x.shape

        # split features by depth axis
        width = int(x.shape[4]/2)

        macula = x[:, :, :, :, :width]
        nerve = x[:, :, :, :, width:]

        B, C, H, W, D = macula.shape

        macula = macula.reshape(B, C, H * W * D).permute(0, 2, 1)
        macula = self.norm1(macula)
        nerve = nerve.reshape(B, C, H * W * D).permute(0, 2, 1)
        nerve = self.norm2(nerve)

        attn_out_1 = self.cross_block1(macula, nerve)
        attn_out_2 = self.cross_block2(nerve, macula)

        attn = x_skip + self.gamma * torch.cat((attn_out_1, attn_out_2), dim=1)

        attn_skip = attn.reshape(xB, xH, xW, xD, xC).permute(
            0, 4, 1, 2, 3)  # (B, C, H, W, D)

        x = attn_skip + self.conv8(attn_skip)

        if self.return_attn:
            mac_nerve_attn = attn_out_1.reshape(
                xB, xH, xW, xD//2, xC).permute(0, 4, 1, 2, 3)
            nerve_mac_attn = attn_out_2.reshape(
                xB, xH, xW, xD//2, xC).permute(0, 4, 1, 2, 3)
            return x, mac_nerve_attn, nerve_mac_attn

        return x


class CrossAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=4, qkv_bias=False, attn_drop_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial)
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size))

    def forward(self, x1, x2):

        B, N, C = x1.shape

        q = self.q(x1).reshape(
            B, N, self.num_heads, C // self.num_heads)
        k = self.k(x2).reshape(
            B, N, self.num_heads, C // self.num_heads)
        v = self.v(x2).reshape(
            B, N, self.num_heads, C // self.num_heads)

        q_x1 = q.permute(0, 2, 1, 3).transpose(-2, -1)
        k_x2 = k.permute(0, 2, 1, 3).transpose(-2, -1)
        v_x2 = v.permute(0, 2, 1, 3).transpose(-2, -1)

        q_x1 = torch.nn.functional.normalize(q_x1, dim=-1)
        k_x2 = torch.nn.functional.normalize(k_x2, dim=-1)

        cross_attn = (q_x1 @ k_x2.transpose(-2, -1)) * self.temperature

        cross_attn = cross_attn.softmax(dim=-1)
        cross_attn = self.attn_drop(cross_attn)

        x_cross = (cross_attn @ v_x2).permute(0, 3, 1, 2).reshape(B, N, C)

        # Projection up
        x_cross = self.out_proj(x_cross)

        return x_cross

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
