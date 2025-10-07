from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rapidfuzz.distance import Levenshtein

from data import CTCShardDataset, ctc_collate
from models import DSConvCTC, DSConvBiGRUCTC
from run_train import greedy_decode, cer_from_batch


CONFIG = {
    "data_root": "./data/shards",  # must contain test/ shards
    "model": "dsconv_bigru",  # "dsconv" or "dsconv_bigru"
    "checkpoint": "./checkpoints/dsconv_bigru_best.pt",
    "batch": 64,
    "num_workers": 2,
}


def make_model(name: str) -> nn.Module:
    if name == "dsconv":
        return DSConvCTC(
            num_symbols=5, width=64, depth=4, kernel=7, p_stem=0.10, p_block=0.10
        )
    if name == "dsconv_bigru":
        return DSConvBiGRUCTC(num_symbols=5, kernel=7, trunk_width=128)
    raise ValueError(f"Unknown model '{name}'")


def load_checkpoint_if_any(model: nn.Module, ckpt_path: str | None):
    if not ckpt_path:
        print("[model] no checkpoint provided. Evaluating random-initialized weights.")
        return
    p = Path(ckpt_path)
    if not p.is_file():
        print(f"[model] checkpoint not found: {p} (skipping)")
        return
    ckpt = torch.load(p, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[model] loaded checkpoint: {p}")


IDX2BASE = {1: "A", 2: "C", 3: "G", 4: "T"}


def tokens_to_str(tokens):
    return "".join(IDX2BASE.get(int(t), "?") for t in tokens)


def show_examples(model, loader, device, max_examples: int = 5):
    """
    Prints a few predictions vs references from the given loader.
    Uses greedy decode. Shows a few base-string samples + per-sample metrics.
    """
    model.eval()
    shown = 0

    with torch.no_grad():
        for X, y_flat, in_lens, tgt_lens in loader:
            X = X.to(device, non_blocking=True)
            y_flat = y_flat.to(device)
            tgt_lens64 = tgt_lens.to(torch.int64)

            logits = model(X)  # [B, T, 5]

            # batch offsets for reference sequences
            y_offsets = torch.zeros(len(tgt_lens64) + 1, dtype=torch.int64)
            y_offsets[1:] = torch.cumsum(tgt_lens64.cpu(), dim=0)

            # greedy predictions
            preds = greedy_decode(logits)

            for i in range(len(preds)):
                if shown >= max_examples:
                    return
                # reference tokens for sample i
                s, e = int(y_offsets[i].item()), int(y_offsets[i + 1].item())
                ref_tokens = y_flat[s:e].cpu().tolist()
                hyp_tokens = preds[i]

                # strings
                ref_str = tokens_to_str(ref_tokens)
                hyp_str = tokens_to_str(hyp_tokens)

                # metrics
                ed = Levenshtein.distance(hyp_tokens, ref_tokens)
                cer = ed / max(1, len(ref_tokens))

                # print
                def trunc(s, n=120):
                    return s if len(s) <= n else s[:n] + "..."

                print(f"\nExample {shown + 1}")
                print(f"  REF: {trunc(ref_str)}")
                print(f"  HYP: {trunc(hyp_str)}")
                print(f"  edit_distance={ed}  cer={cer:.4f}")

                shown += 1
            # only first batch needed
            break


def main():
    cfg = CONFIG
    device = torch.device("cpu")
    print(f"[env] device: {device}")

    # 1) dataset/loader (existing test shards)
    root = Path(cfg["data_root"])
    test_ds = CTCShardDataset(root, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
        collate_fn=ctc_collate,
    )
    print(f"[data] test windows: {len(test_ds)}")

    # 2) model (+ optional checkpoint)
    model = make_model(cfg["model"]).to(device).eval()
    load_checkpoint_if_any(model, cfg.get("checkpoint"))

    # 3) metrics: CTC loss + CER
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss, total_cer, batches = 0.0, 0.0, 0

    with torch.no_grad():
        for X, y_flat, in_lens, tgt_lens in test_loader:
            X = X.to(device, non_blocking=True)
            y_flat = y_flat.to(device)

            in_lens64 = in_lens.to(torch.int64)
            tgt_lens64 = tgt_lens.to(torch.int64)
            out_lens = model.output_lengths(in_lens64)

            logits = model(X)  # [B, T, 5]
            logp = logits.log_softmax(dim=-1).transpose(0, 1).contiguous()  # [T,B,5]
            loss = criterion(logp, y_flat, out_lens, tgt_lens64)

            # batch CER uses offsets derived from tgt lengths
            y_offsets = torch.zeros(len(tgt_lens64) + 1, dtype=torch.int64)
            y_offsets[1:] = torch.cumsum(tgt_lens64.cpu(), dim=0)
            cer = cer_from_batch(logits, y_flat, y_offsets)

            total_loss += float(loss.item())
            total_cer += float(cer)
            batches += 1

    avg_loss = total_loss / max(1, batches)
    avg_cer = total_cer / max(1, batches)
    print(f"[metrics] test_ctc_loss={avg_loss:.4f}  test_cer={avg_cer:.4f}")
    # show a few example comparisons from the same test loader
    show_examples(model, test_loader, device, max_examples=5)


if __name__ == "__main__":
    main()
