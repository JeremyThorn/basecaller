from pathlib import Path
import json
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from generate_synthetic_data import _stable_hash_to_uniform


# One or more CSV paths or directories (dirs are globbed for *.csv)
INPUTS = [
    "./data/final_train_reads",  # directory of CSVs
    # "./some/file.csv",         # and/or specific files
]
OUT_DIR = Path("./data/shards")

# Windowing
WINDOW = 2048
STRIDE = 1024
MIN_TARGETS = 1  # drop windows with fewer than this many target tokens
SHARD_SIZE = 2048  # windows per shard file

# Deterministic split fractions (by filename stem)
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

BASE2IDX = {"A": 1, "C": 2, "G": 3, "T": 4}
BLANK_ID = 0


def assign_split(read_id: str, train_frac: float = 0.8, val_frac: float = 0.1) -> str:
    u = _stable_hash_to_uniform(read_id)
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def robust_norm(x: npt.NDArray) -> tuple[npt.NDArray, np.float32, np.float32]:
    """Median/MAD normalization with mean/std fallback if MAD approx 0."""
    x = x.astype(np.float32, copy=False)
    med = np.median(x).astype(np.float32)
    mad = np.median(np.abs(x - med)).astype(np.float32)
    if mad <= 1e-6:
        mu = np.mean(x).astype(np.float32)
        sd = (np.std(x) + 1e-6).astype(np.float32)
        return (x - mu) / sd, mu, sd
    scale = (mad / 0.6745).astype(np.float32)
    return (x - med) / (scale + 1e-6), med, scale


def window_starts(N: int, window: int, stride: int) -> npt.NDArray:
    if N < window:
        return np.empty((0,), dtype=np.int32)
    return np.arange(0, N - window + 1, stride, dtype=np.int32)


def extract_ctc_targets(idx_seg: npt.NDArray, base_seg: npt.NDArray) -> npt.NDArray:
    """Collapse dwells only (keep true repeats). Map A/C/G/T -> 1..4."""
    if len(idx_seg) == 0:
        return np.empty((0,), dtype=np.int32)
    mask = np.empty(len(idx_seg), dtype=bool)
    mask[0] = True
    mask[1:] = idx_seg[1:] != idx_seg[:-1]
    bases = base_seg[mask].astype(str)
    y = np.fromiter((BASE2IDX[b] for b in bases), count=len(bases), dtype=np.int32)
    return y


def collect_inputs(paths: list[os.PathLike[str]]) -> list[os.PathLike[str]]:
    files: list[os.PathLike[str]] = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            files += sorted(P.glob("*.csv"))
        elif P.suffix == ".csv" and P.is_file():
            files.append(P)
    return files


def write_npz_shard(
    out_path: os.PathLike[str],
    X: npt.NDArray,
    y_flat: npt.NDArray,
    y_offsets: npt.NDArray,
    meta: dict,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32, copy=False),
        y_flat=y_flat.astype(np.int32, copy=False),
        y_offsets=y_offsets.astype(np.int32, copy=False),
        meta=json.dumps(meta),
    )


def main():
    # sanity check: split fractions
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    files = collect_inputs(INPUTS)
    if not files:
        raise SystemExit("No input CSVs found.")

    # prepare out dirs
    for split in ("train", "val", "test"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    # buffers per split
    shard_idx = {"train": 0, "val": 0, "test": 0}
    Xbuf = {"train": [], "val": [], "test": []}
    ybuf = {"train": [], "val": [], "test": []}
    ylen = {"train": [], "val": [], "test": []}

    meta_common = dict(
        version="ctc-minimal-v1",
        window=WINDOW,
        stride=STRIDE,
        normalization="median_mad",
        label_map=BASE2IDX,
        blank_id=BLANK_ID,
    )

    def flush(split: str):
        if not Xbuf[split]:
            return
        X = np.stack(Xbuf[split], axis=0)  # [N, window]
        y_lengths = np.array(ylen[split], dtype=np.int32)
        offsets = np.zeros(len(y_lengths) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(y_lengths, dtype=np.int32)
        y_flat = (
            np.concatenate(ybuf[split], axis=0)
            if offsets[-1] > 0
            else np.empty((0,), dtype=np.int32)
        )

        shard_idx[split] += 1
        out_path = OUT_DIR / split / f"{split}-{shard_idx[split]:05d}.npz"
        meta = {**meta_common, "split": split}
        write_npz_shard(out_path, X, y_flat, offsets, meta)

        Xbuf[split].clear()
        ybuf[split].clear()
        ylen[split].clear()
        print(f"Wrote {out_path} (windows={X.shape[0]})")

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
            continue

        needed = {"current_pA", "ref_base_idx", "ref_base"}
        miss = needed - set(df.columns)
        if miss:
            print(f"{fp} missing columns: {sorted(miss)}")
            continue

        x = df["current_pA"].to_numpy(dtype=np.float32)
        idx = df["ref_base_idx"].to_numpy(dtype=np.int32)
        base = df["ref_base"].astype(str).to_numpy()

        # sanity check: require non-decreasing ref_base_idx (alignments should be monotonic)
        if np.any(idx[1:] < idx[:-1]):
            print(f"{fp}: non-monotonic ref_base_idx. Skipping file.")
            continue

        # per-read normalization
        x, _, _ = robust_norm(x)

        # windowing over raw samples so targets align to true transitions
        starts = window_starts(len(x), WINDOW, STRIDE)
        if len(starts) == 0:  # too short
            continue

        split = assign_split(fp.stem, TRAIN_FRAC, VAL_FRAC)

        for s in starts:
            e = s + WINDOW
            xw = x[s:e]
            idxw = idx[s:e]
            basew = base[s:e]

            y = extract_ctc_targets(idxw, basew)
            if len(y) < MIN_TARGETS:
                continue

            Xbuf[split].append(xw.astype(np.float32, copy=False))
            ybuf[split].append(y.astype(np.int32, copy=False))
            ylen[split].append(len(y))

            if len(Xbuf[split]) >= SHARD_SIZE:
                flush(split)

    for s in ("train", "val", "test"):
        flush(s)

    manifest = dict(
        splits={k: int(v) for k, v in shard_idx.items()}, params=meta_common
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Manifest:", OUT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
