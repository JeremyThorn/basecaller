from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CTCShardDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Dataset for shards produced by prepare_dataset.py.

    Each shard .npz contains:
      - X         : float32 [N, L]         (normalized current windows)
      - y_flat    : int32   [sum_i T_i]    (concatenated targets: A=1,C=2,G=3,T=4)
      - y_offsets : int32   [N+1]          (CSR offsets: y_i = y_flat[offsets[i]:offsets[i+1]])
      - meta      : JSON string (optional, unused here)

    __getitem__(i) returns:
      x      : torch.float32 [L]
      y      : torch.int64   [T]
      x_len  : torch.int32   []   (value L)
      y_len  : torch.int32   []   (value T)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        pattern: str = "*.npz",
        min_targets: int = 1,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.pattern = pattern
        self.min_targets = int(min_targets)

        split_dir = self.root / split
        self.shard_paths: list[Path] = sorted(split_dir.glob(pattern))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shards found in {split_dir} matching {pattern}"
            )

        # Build a flat index of (shard_index, row) for rows with >= min_targets labels.
        self.index: list[tuple[int, int]] = []
        for si, p in enumerate(self.shard_paths):
            with np.load(p, mmap_mode="r", allow_pickle=False) as z:
                y_offsets = z["y_offsets"].astype(np.int64, copy=False)
                y_lengths = np.diff(y_offsets)
            rows = np.nonzero(y_lengths >= self.min_targets)[0]
            self.index.extend((si, int(r)) for r in rows)

        if not self.index:
            raise RuntimeError(
                "No windows with targets found (after min_targets filter)."
            )

        # Single-file cache (last opened shard)
        self._open_si: int | None = None
        self._z: np.lib.npyio.NpzFile | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _ensure_open(self, si: int):
        if self._open_si == si and self._z is not None:
            return
        # Close previous
        if self._z is not None:
            self._z.close()
        # Open requested
        self._z = np.load(self.shard_paths[si], mmap_mode="r", allow_pickle=False)
        self._open_si = si

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        si, row = self.index[idx]
        self._ensure_open(si)
        assert self._z is not None

        X = self._z["X"]  # [N, L]
        y_flat = self._z["y_flat"]  # [sum T]
        y_offsets = self._z["y_offsets"]  # [N+1]

        x_np = X[row]  # [L]
        ys, ye = int(y_offsets[row]), int(y_offsets[row + 1])
        y_np = y_flat[ys:ye]  # [T]

        # Return tensors
        x = torch.as_tensor(x_np, dtype=torch.float32)  # [L]
        y = torch.as_tensor(y_np, dtype=torch.int64)  # [T] (CTC expects Long)
        x_len = torch.tensor(x.shape[0], dtype=torch.int32)
        y_len = torch.tensor(y.shape[0], dtype=torch.int32)

        return x, y, x_len, y_len


def ctc_collate(batch):
    """
    Pads inputs to max length in the batch, concatenates targets, returns lengths.

    Returns:
      X           : float32 [B, L_max]
      y_flat      : int64   [sum T_i]
      input_lens  : int32   [B]
      target_lens : int32   [B]
    """
    xs, ys, xls, yls = zip(*batch)
    B = len(xs)
    Lmax = max(int(x.shape[0]) for x in xs)
    X = torch.zeros(B, Lmax, dtype=torch.float32)
    for i, x in enumerate(xs):
        L = int(x.shape[0])
        X[i, :L] = x
    y_flat = torch.cat(ys, dim=0) if ys else torch.empty((0,), dtype=torch.int64)
    input_lens = torch.stack(xls).to(torch.int32)
    target_lens = torch.stack(yls).to(torch.int32)
    return X, y_flat, input_lens, target_lens
