import torch
from torch import nn
import torch.nn.functional as F


def as_channels(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape [B, 1, L] (accepts [B, L] or [B, 1, L])."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3 and x.size(1) == 1:
        return x
    raise ValueError(f"expected [B, L] or [B, 1, L], got {tuple(x.shape)}")


def same_padding(kernel: int, dilation: int = 1) -> int:
    """Padding that preserves length when stride=1 for odd kernels."""
    return dilation * (kernel - 1) // 2


def conv_out_len(L: torch.Tensor, k: int, s: int, d: int, p: int) -> torch.Tensor:
    """
    Length transform for Conv1d: floor((L + 2p - d*(k-1) - 1)/s + 1).
    Works on tensors of lengths.
    """
    num = L + 2 * p - d * (k - 1) - 1
    return torch.floor_divide(num, s) + 1


def gn_groups(C: int, max_groups: int = 8) -> int:
    """
    Largest group count ≤ max_groups that divides C (stable GroupNorm behavior).
    """
    for g in range(max_groups, 0, -1):
        if C % g == 0:
            return g
    return 1


class DSConvBlock(nn.Module):
    """
    Depthwise-separable conv block:
      - depthwise Conv1d (groups = in_channels)
      - pointwise 1x1 Conv1d
      - GroupNorm with group-size rule
      - SiLU activation

    Stride is applied in the depthwise conv (typical for time downsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = same_padding(kernel_size, dilation)

        self.dw = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # depthwise: C_out = C_in
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,  # pointwise mixes channels
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(gn_groups(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LockedDropoutBF(nn.Module):
    """
    Locked/variational dropout for (B, T, C): one dropout mask per (B, C),
    broadcast across time (T). Keeps temporal structure intact.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # x: [B, T, C],  create (B, 1, C) mask and broadcast over T
        mask = x.new_ones(x.size(0), 1, x.size(2))
        mask = F.dropout(mask, p=self.p, training=True)
        return x * mask


class GRUBlock(nn.Module):
    """
    Residual GRU block (batch-first):
      - Locked dropout on input & output
      - 1-layer GRU
      - Project back to input width if needed for residual add
      - LayerNorm stabilizes residual pathway
    """

    def __init__(
        self,
        width: int,
        hidden: int,
        bidirectional: bool = True,
        p: float = 0.1,
        p_in: None | float = None,
    ):
        super().__init__()
        self.out_dim = hidden * (2 if bidirectional else 1)
        self.in_drop = LockedDropoutBF(p_in if p_in is not None else p)
        self.out_drop = LockedDropoutBF(p)

        self.gru = nn.GRU(
            input_size=width,
            hidden_size=hidden,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.proj = (
            nn.Linear(self.out_dim, width) if self.out_dim != width else nn.Identity()
        )
        self.ln = nn.LayerNorm(width)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, width]
        xin = self.in_drop(x)
        y, h_new = self.gru(xin, h)  # [B, T, out_dim]
        y = self.proj(y)  # [B, T, width]
        y = self.ln(y)
        y = self.out_drop(y)
        return x + y, h_new  # residual


class GRUTwoLayerTrunk(nn.Module):
    """Two residual GRUBlocks with an input projection to `width` if needed."""

    def __init__(
        self,
        input_dim: int,
        width: int = 192,
        bidirectional: bool = True,
        p: float = 0.15,
        p_in: float = 0.2,
    ):
        super().__init__()
        self.in_proj = (
            nn.Linear(input_dim, width) if input_dim != width else nn.Identity()
        )
        self.b1 = GRUBlock(
            width, hidden=width, bidirectional=bidirectional, p=p, p_in=p_in
        )
        self.b2 = GRUBlock(
            width, hidden=width, bidirectional=bidirectional, p=p, p_in=p_in
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x, _ = self.b1(x)
        x, _ = self.b2(x)
        return x
