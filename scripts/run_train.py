from pathlib import Path
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rapidfuzz.distance import Levenshtein

from data import CTCShardDataset, ctc_collate
from models import DSConvCTC, DSConvBiGRUCTC


CONFIG = {
    "data_root": "./data/shards",  # root containing train/ val/ test/
    "model": "dsconv_bigru",  # "dsconv" or "dsconv_bigru"
    "epochs": 50,
    "batch": 64,
    "lr": 2e-3,
    "weight_decay": 1e-2,
    "num_workers": 4,
    "seed": 42,
    "save_path": "./checkpoints/dsconv_bigru_best.pt",
    # Optionally override default restart path (defaults to alongside save_path)
    # "restart_path": "./checkpoints/restart_latest.pt",
    # Model hyperparams
    "kernel": 7,
    # DSConvCTC:
    "width": 64,
    "depth": 4,
    "p_stem": 0.10,
    "p_block": 0.10,
    # DSConvBiGRUCTC:
    "trunk_width": 128,
}


class MetricLogger:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("time,split,epoch,step,metric,value\n")

    def log(self, split: str, epoch: int, step: int, metrics: dict[str, float]):
        ts = f"{time.time():.3f}"
        lines = [
            f"{ts},{split},{epoch},{step},{k},{float(v)}\n" for k, v in metrics.items()
        ]
        with open(self.path, "a", encoding="utf-8") as f:
            f.writelines(lines)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def greedy_decode(logits: torch.Tensor) -> list[list[int]]:
    """Greedy CTC decode (collapse repeats, drop blank=0)."""
    with torch.no_grad():
        pred = logits.argmax(dim=-1)  # [B, T]
        outs: list[list[int]] = []
        for row in pred.cpu().tolist():
            seq, prev = [], None
            for t in row:
                if t == 0:
                    prev = None
                    continue
                if t == prev:
                    continue
                seq.append(t)
                prev = t
            outs.append(seq)
        return outs


def cer_from_batch(
    logits: torch.Tensor, y_flat: torch.Tensor, y_offsets: torch.Tensor
) -> float:
    """Average CER over batch using greedy decode & Levenshtein."""
    preds = greedy_decode(logits)
    total_ed, total_len = 0, 0
    for i, p in enumerate(preds):
        ys, ye = int(y_offsets[i].item()), int(y_offsets[i + 1].item())
        tgt = y_flat[ys:ye].cpu().tolist()
        total_ed += Levenshtein.distance(p, tgt)
        total_len += max(1, len(tgt))
    return total_ed / total_len


def make_model(cfg: dict) -> nn.Module:
    if cfg["model"] == "dsconv":
        return DSConvCTC(
            num_symbols=5,
            width=cfg["width"],
            depth=cfg["depth"],
            kernel=cfg["kernel"],
            p_stem=cfg["p_stem"],
            p_block=cfg["p_block"],
        )
    elif cfg["model"] == "dsconv_bigru":
        return DSConvBiGRUCTC(
            num_symbols=5,
            kernel=cfg["kernel"],
            trunk_width=cfg["trunk_width"],
        )
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")


def default_restart_path(save_path: Path) -> Path:
    return (save_path.parent / "restart_latest.pt").resolve()


def save_restart(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val: float,
    cfg: dict,
):
    """Atomically write a latest-checkpoint suitable for resuming."""
    state = {
        "epoch": int(epoch),  # last completed epoch
        "global_step": int(global_step),
        "best_val": float(best_val),
        "config": cfg,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "py_random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f"[restart] saved latest to {path}")


def maybe_resume(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
):
    """
    If a restart checkpoint exists, load it and return:
      start_epoch, global_step, best_val
    Otherwise returns (1, 0, +inf).
    """
    if not path.exists():
        print(f"[restart] no checkpoint at {path} (fresh run)")
        return 1, 0, float("inf")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    opt.load_state_dict(ckpt["opt_state"])

    # restore RNG states
    try:
        random.setstate(ckpt["py_random_state"])
        np.random.set_state(ckpt["np_random_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])

    except Exception as e:
        print(f"[restart] RNG restore skipped: {e}")

    last_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    start_epoch = last_epoch + 1
    print(
        f"[restart] resumed from {path}: epoch={last_epoch}, step={global_step}, best_val={best_val:.4f}"
    )
    return start_epoch, global_step, best_val


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Datasets / loaders
    root = Path(cfg["data_root"])
    train_ds = CTCShardDataset(root, split="train")
    val_ds = CTCShardDataset(root, split="val")
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=False,
        collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
        collate_fn=ctc_collate,
    )

    # Model / loss / opt
    model = make_model(cfg).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Paths & logger
    save_path = Path(cfg["save_path"]).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    restart_path = Path(
        cfg.get("restart_path", default_restart_path(save_path))
    ).resolve()
    logger = MetricLogger(save_path.with_name("metrics.csv"))

    # Resume (if any)
    start_epoch, global_step, best_val = maybe_resume(restart_path, model, opt, device)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # training
        model.train()
        running = 0.0
        for it, (x, y_flat, in_lens, tgt_lens) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y_flat = y_flat.to(device)

            in_lens64 = in_lens.to(torch.int64)
            tgt_lens64 = tgt_lens.to(torch.int64)
            out_lens = model.output_lengths(in_lens64)  # handle any downsampling

            opt.zero_grad(set_to_none=True)

            logits = model(x)  # [B, T, 5]
            log_probs = (
                logits.log_softmax(dim=-1).transpose(0, 1).contiguous()
            )  # [T, B, 5]
            loss = criterion(log_probs, y_flat, out_lens, tgt_lens64)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            global_step += 1

            # per-iter metrics
            logger.log("train", epoch, global_step, {"ctc_loss": loss.item()})
            if it % 50 == 0:
                print(f"epoch {epoch} iter {it}  train_ctc_loss {running / 50:.4f}")
                running = 0.0

        # validation
        model.eval()
        val_loss = 0.0
        cer_accum = 0.0
        batches = 0
        with torch.no_grad():
            for x, y_flat, in_lens, tgt_lens in val_loader:
                x = x.to(device, non_blocking=True)
                in_lens64 = in_lens.to(torch.int64)
                tgt_lens64 = tgt_lens.to(torch.int64)
                out_lens = model.output_lengths(in_lens64)

                logits = model(x)  # [B, T, 5]
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1).contiguous()
                loss = criterion(log_probs, y_flat, out_lens, tgt_lens64)
                val_loss += loss.item()

                # CER (greedy)
                y_offsets = torch.zeros(len(tgt_lens64) + 1, dtype=torch.int64)
                y_offsets[1:] = torch.cumsum(tgt_lens64.cpu(), dim=0)
                cer = cer_from_batch(logits, y_flat, y_offsets)
                cer_accum += cer
                batches += 1

        val_loss /= max(1, batches)
        cer_avg = cer_accum / max(1, batches)
        print(f"epoch {epoch}  val_ctc_loss {val_loss:.4f}  val_cer {cer_avg:.4f}")

        # epoch-level logging
        logger.log("val", epoch, global_step, {"ctc_loss": val_loss, "cer": cer_avg})

        # save best and rolling latest
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": CONFIG,
                    "best_val_loss": best_val,
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")

        # always refresh restart_latest (after finishing epoch)
        save_restart(
            restart_path,
            model,
            opt,
            epoch=epoch,
            global_step=global_step,
            best_val=best_val,
            cfg=CONFIG,
        )

    print("Done.")


if __name__ == "__main__":
    main()
