#!/usr/bin/env python3
"""Train the M0 multilayer perceptron using frozen ESM-2 embeddings from the training and validation partitions.

Selection is exclusively by validation loss. Test/OOD are never loaded.
Smoke mode performs two optimizer steps on synthetic data, writing nothing.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (SPLIT_HASHES, atomic_json, canonical_json, ids_digest,
                    load_development, require, require_no_partial, sha256, sha_bytes)


class M0Head(nn.Module):
    def __init__(self, widths=(256, 64), dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, widths[0]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths[0], widths[1]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths[1], 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(path, payload):
    stage = path.with_name(path.name + ".partial")
    require(not stage.exists(), f"stale partial checkpoint {stage}")
    torch.save(payload, stage)
    os.replace(stage, path)


def atomic_csv(path, rows):
    stage = path.with_name(path.name + ".partial")
    require(not stage.exists(), f"stale partial history {stage}")
    fields = ["epoch", "train_loss", "validation_loss", "validation_mcc_0_5",
              "validation_f1_0_5", "validation_average_precision", "validation_roc_auc",
              "validation_balanced_accuracy_0_5", "epoch_seconds", "best_epoch_so_far", "epochs_without_improvement"]
    with stage.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, path)


def evaluate(model, data, labels, batch_size, device):
    model.eval()
    probs, loss_sum = [], 0.0
    loss_func = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            features = torch.from_numpy(np.asarray(data[start:end], dtype=np.float32)).to(device)
            targets = torch.from_numpy(labels[start:end].astype(np.float32)).to(device)
            logits = model(features)
            loss_sum += float(loss_func(logits, targets).item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probability = np.concatenate(probs)
    predicted = (probability >= 0.5).astype(int)
    return {
        "loss": loss_sum / len(labels),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root
    home = Path(__file__).resolve().parent
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    require(cfg["model_id"] == "M0" and cfg["hidden_size"] == 1280, "invalid M0 training config")
    config_sha = sha_bytes(canonical_json(cfg))
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M0Head(tuple(cfg["head_widths"]), float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    loss_function = nn.BCEWithLogitsLoss()
    if args.smoke:
        model.train()
        x = torch.randn(8, 1280, device=device)
        y = torch.tensor([0, 1] * 4, dtype=torch.float32, device=device)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x), y)
            require(torch.isfinite(loss).item(), "nonfinite head loss")
            loss.backward()
            optimizer.step()
        print(f"M0 MLP HEAD SMOKE PASS, final synthetic loss={loss.item():.6f}; NO files written")
        return

    frames = load_development(root)  # train and validation ONLY
    meta_path = home / "embeddings/pooled_fp16_train_validation.json"
    npy_path = home / "embeddings/pooled_fp16_train_validation.npy"
    require(meta_path.is_file() and npy_path.is_file(), "M0 embeddings missing: run extract.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require(meta.get("status") == "complete" and meta.get("schema") == "amr_v2_m0_embeddings_v1",
            "M0 embedding metadata is not complete")
    require(meta.get("config_sha256") == config_sha and meta.get("frozen_train_validation_sha256") == SPLIT_HASHES,
            "M0 embedding config/source hashes differ")
    require(meta.get("partition_order") == ["train", "validation"], "unexpected embedding order")
    require(meta.get("partition_rows") == {s: len(frames[s]) for s in ("train", "validation")},
            "unexpected embedding partition counts")
    require(meta.get("partition_ids_sha256") == {s: ids_digest(frames[s]) for s in ("train", "validation")},
            "embedding IDs do not match training and validation partitions")
    combined_ids = __import__("pandas").concat([frames["train"], frames["validation"]], ignore_index=True)
    require(meta.get("all_ids_sha256") == ids_digest(combined_ids), "embedding row ordering mismatch")
    require(sha256(npy_path) == meta.get("file_sha256"), "embedding file bytes changed")
    values = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    require(values.shape == (sum(len(frames[s]) for s in ("train", "validation")), 1280) and values.dtype == np.float16,
            "embedding shape/dtype mismatch")
    require(np.isfinite(values).all(), "embedding array contains non-finite values")
    train_count = len(frames["train"])
    x_train = values[:train_count]
    x_val = values[train_count:]
    y_train = frames["train"]["label"].to_numpy(dtype=np.int64)
    y_val = frames["validation"]["label"].to_numpy(dtype=np.int64)
    print(f"Verified M0 embeddings: train {len(y_train):,}; validation {len(y_val):,}", flush=True)
    print("Checkpoint selection: validation loss only; test/OOD not loaded.", flush=True)

    directory = home / "runs" / f"seed{seed}"
    require_no_partial(directory)
    directory.mkdir(parents=True, exist_ok=True)
    best = directory / "best_model.pt"
    last = directory / "last_checkpoint.pt"
    history_path = directory / "history.csv"
    summary = directory / "training_summary.json"
    config_copy = directory / "config.json"
    require(not summary.exists(), "M0 run already complete; refusing to overwrite")
    if config_copy.exists():
        require(json.loads(config_copy.read_text(encoding="utf-8")) == cfg, "run config differs")
    else:
        require(not best.exists() and not last.exists() and not history_path.exists(),
                "orphaned M0 output files without config")
        atomic_json(config_copy, cfg)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_history = []
    best_loss = float("inf")
    best_epoch, patience_count, start_epoch = 0, 0, 1
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        require(state.get("config_sha256") == config_sha and
                state.get("embedding_sha256") == meta["file_sha256"],
                "resume checkpoint refers to another experiment/data")
        require(best.exists(), "resume checkpoint present but best model missing")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        patience_count = int(state["patience_count"])
        start_epoch = int(state["epoch"]) + 1
        all_history = list(state["history"])
        require(len(all_history) == start_epoch - 1, "resume history length mismatch")
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming run at epoch {start_epoch}, best {best_epoch}")
    else:
        require(not best.exists() and not history_path.exists(),
                "partial run without last_checkpoint.pt; do not overwrite")

    start_time = time.time()
    stop_reason = "max_epochs"
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["early_stopping_patience"])
    batch_size = int(cfg["batch_size"])
    for epoch in range(start_epoch, max_epochs + 1):
        if patience_count >= patience:
            stop_reason = "early_stopping"
            break
        start = time.time()
        model.train()
        order = torch.randperm(len(y_train), generator=generator).numpy()
        total_loss = 0.0
        for offset in range(0, len(order), batch_size):
            idx = order[offset:offset + batch_size]
            batch_x = torch.from_numpy(np.asarray(x_train[idx], dtype=np.float32)).to(device)
            batch_y = torch.from_numpy(y_train[idx].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            train_loss = loss_function(model(batch_x), batch_y)
            require(torch.isfinite(train_loss).item(), "nonfinite training loss")
            train_loss.backward()
            optimizer.step()
            total_loss += float(train_loss.item()) * len(idx)
        val = evaluate(model, x_val, y_val, batch_size, device)
        require(np.isfinite(val["loss"]), "nonfinite validation loss")
        if val["loss"] < best_loss - float(cfg["early_stopping_min_delta"]):
            best_loss = float(val["loss"])
            best_epoch = epoch
            patience_count = 0
            atomic_checkpoint(best, {
                "model": model.state_dict(), "epoch": epoch, "validation_loss": best_loss,
                "config_sha256": config_sha, "embedding_sha256": meta["file_sha256"],
                "backbone_commit": meta["backbone_commit"],
            })
        else:
            patience_count += 1
        all_history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / len(y_train), 8),
            "validation_loss": round(val["loss"], 8),
            "validation_mcc_0_5": round(val["mcc"], 6),
            "validation_f1_0_5": round(val["f1"], 6),
            "validation_average_precision": round(val["average_precision"], 6),
            "validation_roc_auc": round(val["roc_auc"], 6),
            "validation_balanced_accuracy_0_5": round(val["balanced_accuracy"], 6),
            "epoch_seconds": round(time.time() - start, 3),
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": patience_count,
        })
        atomic_checkpoint(last, {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_loss": best_loss, "best_epoch": best_epoch, "patience_count": patience_count,
            "history": all_history, "config_sha256": config_sha,
            "embedding_sha256": meta["file_sha256"],
            "sampler_rng": generator.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        atomic_csv(history_path, all_history)
        print(f"Epoch {epoch:02d}/{max_epochs}: train_loss={all_history[-1]['train_loss']:.5f}, "
              f"val_loss={val['loss']:.5f}, val_MCC={val['mcc']:.4f}, "
              f"best={best_epoch}, unimproved={patience_count}", flush=True)
        if patience_count >= patience:
            stop_reason = "early_stopping"
            break

    require(best.is_file() and last.is_file() and all_history,
            "M0 training yielded no checkpoint/history")
    result = {
        "model_id": "M0", "dataset_version": "V2", "status": "complete",
        "seed": seed, "epochs_executed": len(all_history), "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "stop_reason": stop_reason,
        "checkpoint_metric": "validation_loss", "selection_uses_test_or_ood": False,
        "train_rows": len(y_train), "validation_rows": len(y_val),
        "config_sha256": config_sha, "embedding_sha256": meta["file_sha256"],
        "backbone_commit": meta["backbone_commit"],
        "best_model_sha256": sha256(best), "last_checkpoint_sha256": sha256(last),
        "elapsed_seconds_this_invocation": round(time.time() - start_time, 3),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }
    atomic_json(summary, result)
    print("M0 TRAINING COMPLETE; best epoch:", best_epoch)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
