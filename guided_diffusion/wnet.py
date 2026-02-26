from abc import abstractmethod

import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3 and self.resample_2d:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            num_groups=32,
            resample_2d=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_groups = num_groups

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class WNetModel(nn.Module):
    """
    完整的 W-Net 模型 (U-Net + U-Net)，包含 Decoder1 -> Encoder2 的 Dense Skip Connections。
    修复了通道数不匹配、变量覆盖和硬编码问题。
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            num_groups=32,
            bottleneck_attention=True,
            resample_2d=True,
            additive_skips=False,
            decoder_device_thresh=0
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.devices = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ============================================================
        # U-Net 1: Encoder (Input Blocks)
        # ============================================================
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.input_block_chans_bk = input_block_chans[:]

        # ============================================================
        # U-Net 1: Middle Block
        # ============================================================
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
            *([AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            )] if self.bottleneck_attention else [])
            ,
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
        )
        self._feature_size += ch
        unet1_mid_out_ch = ch
        # ============================================================
        # U-Net 1: Decoder (Output Blocks)
        # ============================================================
        self.output_blocks = nn.ModuleList([])

        # [NEW] 用于记录 Decoder1 每一层的输出通道，供 Encoder2 拼接使用
        self.dec1_out_chans = []

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                mid_ch = model_channels * mult if not self.additive_skips else (
                    input_block_chans[-1] if input_block_chans else model_channels
                )

                # ResBlock (Upsampling path)
                layers = [
                    ResBlock(
                        ch + ich if not self.additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                ch = mid_ch
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d
                        )
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

                # [NEW] 记录当前层输出通道
                self.dec1_out_chans.append(ch)
                mid_ch = ch

        # U-Net 1 Output
        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            # [FIXED] 使用 out_channels 而不是硬编码 3
            zero_module(conv_nd(dims, model_channels, 1, 3, padding=1)),
        )

        # ============================================================
        # U-Net 2: Encoder (Input Blocks 1) - With Skip Connections
        # ============================================================
        # [FIXED] 输入通道应该是上一个网络的输出 out_channels
        mid_channels = 1
        self.input_blocks_1 = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, mid_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans_1 = [model_channels]
        ch = model_channels
        ds = 1

        # Decoder 1 输出的通道数列表
        skip_chans_source = self.dec1_out_chans[::-1]
        skip_idx = 0

        # [NEW] 定义 Bridge 层列表，用于调整 Skip Connection 的维度和分辨率
        self.skip_bridges = nn.ModuleList()

        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                # 获取源通道数
                dec1_skip_ch = skip_chans_source[skip_idx]
                skip_idx += 1

                # [关键逻辑] 确定 Stride
                # Level 0: 始终匹配，Stride=1
                # Level > 0: 只有第一个 Block 会遇到 Decoder 的 Upsample 输出 (大图)，需要 Stride=2
                if level > 0 and i == 0:
                    # 我们在进入新 Level 的第一层需要下采样
                    if resample_2d and dims == 3:
                        # 3D 数据但只做 2D 下采样 -> stride=(1, 2, 2)
                        cur_stride = (1, 2, 2)
                    else:
                        # 普通情况 -> stride=2 (全维度下采样)
                        cur_stride = 2
                else:
                    cur_stride = 1

                # 构建 Bridge 卷积:
                # 作用 1: 调整分辨率 (通过 stride)
                # 作用 2: 统一特征空间 (可学习)
                bridge = conv_nd(
                    dims,
                    dec1_skip_ch,
                    dec1_skip_ch,  # 输出通道保持不变，或者你也可以在这里降维
                    3,
                    stride=cur_stride,
                    padding=1
                )
                self.skip_bridges.append(bridge)

                # 构建 ResBlock
                layers = [
                    ResBlock(
                        channels=ch + dec1_skip_ch,  # Bridge 后的通道数应与此匹配
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                self.input_blocks_1.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans_1.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch

                # 处理 Downsample 层的 Skip
                dec1_skip_ch = skip_chans_source[skip_idx]
                skip_idx += 1

                # Downsample 层对应的通常是 Decoder 中间层，不需要 stride 2
                bridge = conv_nd(dims, dec1_skip_ch, dec1_skip_ch, 3, stride=1, padding=1)
                self.skip_bridges.append(bridge)

                self.input_blocks_1.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels=ch + dec1_skip_ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Downsample(
                            ch + dec1_skip_ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d,
                        )
                    )
                )
                ch = out_ch
                input_block_chans_1.append(ch)
                ds *= 2
                self._feature_size += ch

        self.input_block_chans_bk_1 = input_block_chans_1[:]

        # 确保消耗了所有 skip channels (减去起始卷积不用的)
        # assert len(skip_chans_source) - skip_idx <= 1, \
        #     f"Skip channels mismatch: consumed {skip_idx}, available {len(skip_chans_source)}"

        # ============================================================
        # U-Net 2: Middle Block
        # ============================================================
        self.middle_block_1 = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
            *([AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            )] if self.bottleneck_attention else [])
            ,
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
        )
        self._feature_size += ch
        # ch = ch + unet1_mid_out_ch
        # ============================================================
        # U-Net 2: Decoder (Output Blocks 1)
        # ============================================================
        self.output_blocks_1 = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans_1.pop()
                mid_ch = model_channels * mult if not self.additive_skips else (
                    input_block_chans_1[-1] if input_block_chans_1 else model_channels
                )

                layers = [
                    ResBlock(
                        ch + ich if not self.additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                ch = mid_ch
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d
                        )
                    )
                    ds //= 2
                self.output_blocks_1.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                mid_ch = ch

        self.out_1 = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def to(self, *args, **kwargs):
        """
        优化后的显存分配逻辑：
        Device 0: 负责整个 U-Net 1 (Coarse Stage)
        Device 1: 负责整个 U-Net 2 (Refinement Stage)
        """
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            self.devices = args[0]

            print(f"Distributing W-Net Strategy: [U-Net 1 -> {self.devices[0]}] | [U-Net 2 -> {self.devices[1]}]")

            # ============================================
            # 1. 配置 Device 0 (U-Net 1 全家桶)
            # ============================================
            dev0 = self.devices[0]

            # 公共 Embedding 层通常放在主卡，forward 时传给从卡
            self.time_embed.to(dev0)
            if hasattr(self, 'label_emb') and self.label_emb is not None:
                self.label_emb.to(dev0)

            # U-Net 1 的所有组件
            self.input_blocks.to(dev0)
            self.middle_block.to(dev0)
            self.output_blocks.to(dev0)  # 整个 Decoder 1 都在 dev0
            self.out.to(dev0)

            # ============================================
            # 2. 配置 Device 1 (U-Net 2 全家桶)
            # ============================================
            dev1 = self.devices[1]

            # U-Net 2 的所有组件
            self.input_blocks_1.to(dev1)
            self.middle_block_1.to(dev1)

            # Skip Bridges 连接层放在目标设备 (Dev 1) 上
            # 这样数据从 Dev 0 传过来后，在 Dev 1 上进行卷积处理
            if hasattr(self, 'skip_bridges'):
                self.skip_bridges.to(dev1)

            self.output_blocks_1.to(dev1)  # 整个 Decoder 2 都在 dev1
            self.out_1.to(dev1)

            print(f"Distributed successfully: Split model between {dev0} and {dev1}")

        else:
            # 单卡模式
            super().to(*args, **kwargs)
            if self.devices is None:
                p = next(self.parameters())
                self.devices = [p.device, p.device]

    def forward_unet1(self, x, timesteps, y=None):
        """
        仅执行 U-Net 1 的前向传播，用于 Stage 1 预训练
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            new_hs = hs.pop()
            # 自动对齐逻辑 (保持你现有的)
            if h.shape[2:] != new_hs.shape[2:]:
                new_hs = F.interpolate(new_hs, size=h.shape[2:], mode="nearest")

            if self.additive_skips:
                h = (h + new_hs) / 2
            else:
                h = th.cat([h, new_hs], dim=1)

            h = module(h, emb)

        return self.out(h)
    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # 初始化设备检查
        if self.devices is None:
            # 如果没有调用过 to()，尝试自动推断
            self.devices = [x.device, x.device]
        # print(f"当前设备: {x.device}")
        # print(f"当前设备: {self.devices}")
        assert x.device == self.devices[0]

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # ==========================================
        # 1. U-Net 1 Forward
        # ==========================================
        h = x
        hs = []
        # Encoder 1
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # Middle 1
        h = self.middle_block(h, emb)
        # h_se = h.to(self.devices[1])
        # Decoder 1
        h_dec1_outputs = []  # [NEW] 收集 Decoder 1 的特征

        for k, module in enumerate(self.output_blocks):
            new_hs = hs.pop()

            # print(f"Decoder1 {k} input shape: {new_hs.shape}")
            # Skip connection (Enc1 -> Dec1)
            if self.additive_skips:
                h = (h + new_hs) / 2
            else:
                h = th.cat([h, new_hs], dim=1)

            h = module(h, emb)

            # [NEW] 保存特征用于 U-Net 2 (注意：这里还没经过 out 层，特征丰富)
            # print(f"Decoder1 {k} output shape: {h.shape}")
            h_dec1_outputs.append(h)

        # U-Net 1 Output
        h_mid = h.to(self.devices[0])
        w_m = self.out(h_mid)

        # ==========================================
        # 2. U-Net 2 Forward (With Skip from Dec1)
        # ==========================================
        # [FIXED] 确保 emb 回到 device 0
        emb = emb.to(self.devices[1])
        h_s = w_m.to(self.devices[1])
        h_down = []

        dec1_skips = h_dec1_outputs[::-1]

        # 1. Initial Conv
        h_s = self.input_blocks_1[0](h_s, emb)
        h_down.append(h_s)

        # 2. Loop over blocks
        skip_idx = 0

        for i, module in enumerate(self.input_blocks_1[1:]):

            # 获取 Decoder 1 特征
            skip_feat = dec1_skips[skip_idx]

            # 获取对应的 Bridge Layer
            bridge = self.skip_bridges[skip_idx]
            skip_idx += 1

            # 设备对齐
            if skip_feat.device != self.devices[1]:
                skip_feat = skip_feat.to(self.devices[1])
            skip_feat = bridge(skip_feat)
            h_s = th.cat([h_s, skip_feat], dim=1)
            h_s = module(h_s, emb)
            h_down.append(h_s)

        # Middle 2
        h_s = self.middle_block_1(h_s, emb)
        # h_s = th.cat([h_s, h_se], dim=1)
        # print(f"Middle2 input shape: {h_s.shape}")
        # Decoder 2 (Standard U-Net logic)
        for k, module in enumerate(self.output_blocks_1):
            new_hs = h_down.pop()
            # print(f"Decoder2 {k} input shape: {new_hs.shape}")
            if self.additive_skips:
                # [FIXED] 智能处理通道不匹配的情况
                if h_s.shape[1] == new_hs.shape[1]:
                    # 情况 A: 通道数完全匹配（通常是后续的层），直接相加
                    h_s = (h_s + new_hs) / 2
                else:
                    # 情况 B: 通道数不匹配（发生在第一层，因为 h_s 拼接了 h_se）
                    # h_s 的结构是 [当前流特征, U-Net1特征]
                    # new_hs 对应的是 [当前流特征]
                    # 解决方案：只将 skip 加到前半部分，后半部分保持原样拼接
                    channels = new_hs.shape[1]

                    # 确保 h_s 足够大，包含了 channels
                    assert h_s.shape[1] > channels, "Feature map is smaller than skip connection, logic error."

                    h_s_main = h_s[:, :channels]  # 取出前半部分
                    h_s_extra = h_s[:, channels:]  # 取出后半部分 (U-Net 1 的特征)

                    # 加和前半部分，再拼回去
                    h_s_combined = (h_s_main + new_hs) / 2
                    h_s = th.cat([h_s_combined, h_s_extra], dim=1)
            else:
                h_s = th.cat([h_s, new_hs], dim=1)
            h_s = module(h_s, emb)

        # [FIXED] 最终输出处理，不要用 h (那是第一个网的变量)
        h_final = h_s
        out_final = self.out_1(h_final)

        return w_m, out_final.to(self.devices[0])

if __name__ == '__main__':
    model = WNetModel(image_size=32,
                      in_channels=4,
                      out_channels=8,
                      model_channels=64,
                      num_res_blocks=2,
                      num_heads=4,
                      attention_resolutions=[8],
                      dims=3,
                      # channel_mult=(1, 1, 2, 3, 4),
                      additive_skips=True,
                      )
    # model.to(torch.device('cuda:0'))
    x = torch.randn(1, 4, 128, 128, 64)
    timesteps = torch.randint(0, 1000, (1,))
    print(timesteps)
    f, output = model(x, timesteps=timesteps)
    print(f.shape, output.shape)
