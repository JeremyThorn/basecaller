import torch
from torch import nn

from blocks import *


class DSConvCTC(nn.Module):
    """
    Simple same-length encoder, conv CTC head.

    Stem:
      - Conv1d(1->width, stride=1, 'same' pad) + SiLU + GroupNorm + Dropout1d
    Blocks:
      - depthwise-separable conv blocks (stride=1, 'same' pad)
    Head:
      - 1x1 Conv to num_symbols

    No time downsampling: T_out == T_in.
    """

    def __init__(
        self,
        num_symbols: int = 5,
        width: int = 64,
        depth: int = 4,
        kernel: int = 7,
        p_stem: float = 0.10,  # dropout after the stem
        p_block: float = 0.10,  # dropout between DSConv blocks
    ):
        super().__init__()
        pad = same_padding(kernel, 1)

        # stem: lift 1 channel to `width` with a plain conv
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=kernel, padding=pad, bias=False),
            nn.SiLU(),
            nn.GroupNorm(gn_groups(width), width),
            nn.Dropout1d(p_stem) if p_stem > 0 else nn.Identity(),
        )

        # DSConv stack (stride=1, preserves length)
        blocks: list[nn.Module] = []
        for _ in range(depth):
            blocks.append(
                DSConvBlock(width, width, kernel_size=kernel, stride=1, dilation=1)
            )
            if p_block > 0:
                blocks.append(nn.Dropout1d(p_block))
        self.encoder = nn.Sequential(*blocks)

        # classification head
        self.head = nn.Conv1d(width, num_symbols, kernel_size=1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # Stem + DSConv blocks use stride=1 with 'same' padding, length preserved.
        return input_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = as_channels(x)  # [B,1,L]
        x = self.stem(x)  # [B,width,L]
        x = self.encoder(x)  # [B,width,L]
        x = self.head(x)  # [B,num_symbols,L]
        return x.transpose(1, 2)  # [B,L,num_symbols]


class DSConvBiGRUCTC(nn.Module):
    """
    Strided DSConv encoder (x/4) + 2-layer residual GRU trunk + linear CTC head.

    Encoder:
      - Conv stem (regular conv) + Dropout1d
      - DSConv stride-2  → /2
      - DSConv stride-2  → /4
      - DSConv stride-1 (dilation=1)
      - DSConv stride-1 (dilation=2), same-length thanks to 'same' padding

    Trunk:
      - GRU x2 with locked dropout, residual, LayerNorm

    Head:
      - Linear over features per time step
    """

    def __init__(
        self,
        num_symbols: int = 5,
        initial_channels: int = 16,
        trunk_width: int = 128,
        bidirectional: bool = True,
        kernel: int = 7,
        p_stem: float = 0.075,
        p_stride: float = 0.15,
        p_tail: float = 0.075,
    ):
        super().__init__()
        pad = same_padding(kernel, 1)

        # Simple stem conv to lift from 1 channel
        self.stem = nn.Sequential(
            nn.Conv1d(1, initial_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.SiLU(),
            nn.GroupNorm(gn_groups(initial_channels), initial_channels),
            nn.Dropout1d(p_stem),
        )

        # Downsampling path (depthwise separable)
        self.ds1 = DSConvBlock(
            initial_channels, 32, kernel_size=kernel, stride=2, dilation=1
        )  # /2
        self.drop1 = nn.Dropout1d(p_stride)
        self.ds2 = DSConvBlock(32, 64, kernel_size=kernel, stride=2, dilation=1)  # /4
        self.drop2 = nn.Dropout1d(p_stride)

        # Same-length refiners
        self.ds3 = DSConvBlock(64, 96, kernel_size=kernel, stride=1, dilation=1)
        self.ds4 = DSConvBlock(96, 128, kernel_size=kernel, stride=1, dilation=2)
        self.tail_drop = nn.Dropout1d(p_tail)

        # GRU trunk expects features as [B, T, C]
        self.trunk = GRUTwoLayerTrunk(
            input_dim=128,
            width=trunk_width,
            bidirectional=bidirectional,
            p=0.15,
            p_in=0.20,
        )
        self.head = nn.Linear(trunk_width, num_symbols)

        # Save conv hyperparams used for exact length computation
        self._conv_len_spec = [
            # (k, s, d, p)
            (kernel, 1, 1, pad),  # stem
            (kernel, 2, 1, same_padding(kernel, 1)),  # ds1 depthwise stride-2
            (kernel, 2, 1, same_padding(kernel, 1)),  # ds2 depthwise stride-2
            (kernel, 1, 1, same_padding(kernel, 1)),  # ds3
            (kernel, 1, 2, same_padding(kernel, 2)),  # ds4 (dilated, stride-1)
        ]

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Exact Conv1d length transform through the encoder.
        """
        L = input_lengths.to(torch.int64)
        for k, s, d, p in self._conv_len_spec:
            L = conv_out_len(L, k=k, s=s, d=d, p=p)
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = as_channels(x)  # [B,1,L]
        x = self.stem(x)  # [B,C,~L]
        x = self.ds1(x)
        x = self.drop1(x)  # [B,32,L/2]
        x = self.ds2(x)
        x = self.drop2(x)  # [B,64,L/4]
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.tail_drop(x)

        # To GRU: [B, T, C]
        x = x.transpose(1, 2).contiguous()
        x = self.trunk(x)  # [B,T,trunk_width]
        return self.head(x)  # [B,T,num_symbols]
