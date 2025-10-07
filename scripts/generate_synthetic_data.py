import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import itertools
import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from tslearn.metrics import dtw_path_from_metric

# Stop BLAS from oversubscribing when we fan out workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def generate_kmers(bases: str, k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def _stable_hash_to_uniform(s: str) -> float:
    """Deterministic uniform in [0,1)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big", signed=False)
    return x / 2**64


def build_kmer_model(
    bases: str = "ACGT",
    k: int = 5,
    current_center: float = 70.0,
    pos_scale_center: float = 2.8,
    pos_scale_flank: float = 2.6,
    pair_scale: float = 0.60,
    sd_lo: float = 0.60,
    sd_hi: float = 0.95,
    global_jitter: float = 0.02,
) -> dict[str, tuple[float, float]]:
    """
    Returns a simple dict: { kmer: (mean, sd) }.
    """
    if k % 2 == 0:
        raise ValueError("k must be odd (unique center).")
    if not (sd_lo > 0 < sd_hi and sd_hi > sd_lo):
        raise ValueError("Require 0 < sd_lo < sd_hi.")

    kmers = generate_kmers(bases, k)
    mid = k // 2
    nB = len(bases)

    # Center levels: equally spaced, zero-mean.
    raw_levels = np.linspace(-(nB - 1) / 2.0, (nB - 1) / 2.0, nB) * pos_scale_center
    base_order = sorted(
        list(bases), key=lambda b: _stable_hash_to_uniform(f"center_order:{b}")
    )
    center_levels = {b: float(raw_levels[i]) for i, b in enumerate(base_order)}

    # Per-position contributions
    pos_contrib = {}
    for p in range(k):
        for b in bases:
            if p == mid:
                pos_contrib[(p, b)] = center_levels[b]
            else:
                u = _stable_hash_to_uniform(f"pos:{p}:{b}")
                pos_contrib[(p, b)] = (u - 0.5) * pos_scale_flank

    # Adjacent-pair contributions
    pair_contrib = {}
    for p in range(k - 1):
        for b1 in bases:
            for b2 in bases:
                u = _stable_hash_to_uniform(f"pair:{p}:{b1}{b2}")
                pair_contrib[(p, b1, b2)] = (u - 0.5) * pair_scale

    # Assemble table
    table: dict[str, tuple[float, float]] = {}
    for kmer in kmers:
        m = current_center
        for p in range(k):
            m += pos_contrib[(p, kmer[p])]
        for p in range(k - 1):
            m += pair_contrib[(p, kmer[p], kmer[p + 1])]
        u_sd = _stable_hash_to_uniform("sd:" + kmer)
        sd = sd_lo + (sd_hi - sd_lo) * u_sd
        if global_jitter:
            m += (u_sd - 0.5) * global_jitter
        table[kmer] = (float(m), float(sd))
    return table


def generate_sequence(
    length: int, bases: str = "ACGT", rng: np.random.Generator | None = None
) -> str:
    rng = rng or np.random.default_rng()
    idx = rng.integers(0, len(bases), length)
    return "".join(bases[i] for i in idx)


def kmers_from_sequence(sequence: str, k: int) -> list[str]:
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def current_dist_from_kmers(
    kmers: list[str], model_table: dict[str, tuple[float, float]]
) -> npt.NDArray[np.float64]:
    return np.array([model_table[k] for k in kmers], dtype=float)


def generate_current_series(
    kmers: list[str],
    model_table: dict[str, tuple[float, float]],
    median_dwell: float = 11.0,
    dwell_lognorm_sigma: float = 0.65,
    white_noise: float = 0.12,
    sample_rate: float = 4000.0,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = rng or np.random.default_rng()
    mean_stds = current_dist_from_kmers(kmers, model_table)  # (L, 2)

    mu = np.log(median_dwell)
    dwell = rng.lognormal(mu, dwell_lognorm_sigma, size=mean_stds.shape[0])
    dwell = np.maximum(np.rint(dwell), 1).astype(int)

    means_stds = np.repeat(mean_stds, dwell, axis=0)
    drift = np.arange(means_stds.shape[0]) * 1e-6 * rng.uniform(0.8, 1.2)

    dt = 1.0 / sample_rate
    t = np.arange(means_stds.shape[0], dtype=np.float64) * dt

    current = rng.normal(
        drift + means_stds[:, 0], np.sqrt(means_stds[:, 1] ** 2 + white_noise**2)
    )
    return t, current


def dtw_align_banded(
    s: npt.NDArray[np.float64], t: npt.NDArray[np.float64], band_frac: float = 0.12
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Indices mapping s (signal) to t (template) via Sakoe–Chiba band DTW."""
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    N, M = len(s), len(t)
    if N == 0 or M == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    base_r = int(np.ceil(band_frac * M))
    radius = max(base_r, abs(N - M))
    path, _ = dtw_path_from_metric(
        s,
        t,
        metric="sqeuclidean",
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=radius,
    )
    i_idx, j_idx = map(np.array, zip(*path))
    return i_idx.astype(int), j_idx.astype(int)


def generate_one_read(
    seed: int,
    model_table: dict[str, tuple[float, float]],
    bases: str,
    k: int,
    out_dir: os.PathLike[str],
    median_dwell: float,
    dwell_lognorm_sigma: float,
    extra_white_noise: float,
) -> int:
    rng = np.random.default_rng(int(seed))
    seq = generate_sequence(1000, bases=bases, rng=rng)
    kmers = kmers_from_sequence(seq, k)

    current_template = current_dist_from_kmers(kmers, model_table)[:, 0]
    t, A = generate_current_series(
        kmers,
        model_table,
        median_dwell=median_dwell,
        dwell_lognorm_sigma=dwell_lognorm_sigma,
        white_noise=extra_white_noise,
        rng=rng,
    )

    sig_idx, tmp_idx = dtw_align_banded(A, current_template, band_frac=0.12)
    ref_base_pos = np.clip(tmp_idx + (k // 2), 0, len(seq) - 1)

    df = pd.DataFrame(
        {
            "t_sec": t[sig_idx].astype("float32"),
            "current_pA": A[sig_idx].astype("float32"),
            "ref_pos": tmp_idx.astype("int32"),
            "ref_kmer": [kmers[j] for j in tmp_idx],
            "ref_base_idx": ref_base_pos.astype("int32"),
            "ref_base": [seq[i] for i in ref_base_pos],
        }
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"read_V8_{seed}.csv"
    df.to_csv(out_path, index=False)
    return int(seed)


if __name__ == "__main__":
    bases = "ACGT"
    k = 5

    model_table = build_kmer_model(
        bases=bases,
        k=k,
        current_center=70.0,
        pos_scale_center=2.8,
        pos_scale_flank=2.6,
        pair_scale=0.60,
        sd_lo=0.60,
        sd_hi=0.95,
        global_jitter=0.02,
    )

    seeds = np.arange(0, 800)
    median_dwell = 11.0
    dwell_lognorm_sigma = 0.65
    extra_white_noise = 0.12
    out_dir = Path("./data/final_train_reads")

    fn = functools.partial(
        generate_one_read,
        model_table=model_table,
        bases=bases,
        k=k,
        out_dir=out_dir,
        median_dwell=median_dwell,
        dwell_lognorm_sigma=dwell_lognorm_sigma,
        extra_white_noise=extra_white_noise,
    )

    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, int(s)) for s in seeds]
        for fut in as_completed(futures):
            _ = fut.result()  # re-raise on worker error
