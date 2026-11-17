#!/usr/bin/env python3
"""Extract M4 pooled ProtBERT embeddings from training and validation data only.

Explicit stages: --smoke runs only two proteins and writes nothing;
full extraction uses one separate cache and a safe resume cursor on Google Drive.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (SPLIT_HASHES, atomic_json, canonical_json, ids_digest,
                    load_development, require, require_no_partial, sha256, sha_bytes)


def load_encoder(config):
    require(torch.cuda.is_available(), "CUDA GPU required for ProtBERT extraction")
    require(torch.cuda.is_bf16_supported(), "GPU BF16 required; no automatic precision fallback")
    name, commit = config["backbone"], config["backbone_revision"]
    require(name == "Rostlab/prot_bert", "unexpected ProtBERT backbone")
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit.lower()),
            "ProtBERT commit not pinned; run pin_backbone.py before extraction")
    from transformers import BertModel, BertTokenizer
    print(f"Loading {name} from pinned revision {commit} with BF16...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(name, revision=commit, do_lower_case=False)
    # add_pooling_layer=False excludes the pretrained checkpoint's absent CLS pooler;
    # the validated representation is a residue-wise mean, not CLS pooling.
    model = BertModel.from_pretrained(name, revision=commit, add_pooling_layer=False, dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to("cuda")
    require(int(model.config.hidden_size) == 1024, "wrong ProtBERT hidden size")
    print("Model repository commit:", commit, "| GPU:", torch.cuda.get_device_name(0), flush=True)
    return tokenizer, model, commit


def prep_sequence(sequence):
    # Official ProtBERT format: uppercase letters separated by spaces.
    sequence = str(sequence).upper()
    sequence = sequence.translate(str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"}))
    require(set(sequence) <= set("ACDEFGHIKLMNPQRSTVWYX"), "unexpected amino acid")
    return " ".join(sequence)


def encode_batch(sequences, tokenizer, model):
    spaced = [prep_sequence(s) for s in sequences]
    payload = tokenizer(spaced, return_tensors="pt", padding=True,
                        truncation=False, return_special_tokens_mask=True)
    input_ids = payload["input_ids"].to("cuda")
    attention = payload["attention_mask"].to("cuda")
    special = payload["special_tokens_mask"].to("cuda")
    require(input_ids.shape[1] <= model.config.max_position_embeddings,
            "ProtBERT positional limit exceeded")
    residue = attention.bool() & ~special.bool()
    observed = residue.sum(dim=1).cpu().tolist()
    expected = [len(s) for s in sequences]
    require(observed == expected, "ProtBERT tokenization lost or introduced residues")
    require(not bool((input_ids[residue] == tokenizer.unk_token_id).any().item()),
            "ProtBERT produced [UNK] residue tokens")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(input_ids=input_ids, attention_mask=attention).last_hidden_state
        mask = residue.unsqueeze(-1)
        vectors = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
    output = vectors.cpu().numpy().astype(np.float16)
    require(output.shape == (len(sequences), 1024) and np.isfinite(output).all(),
            "ProtBERT pooled embedding shape/value problem")
    return output


def plan_batch(order, offset, lengths, max_sequences, token_budget):
    positions = []
    longest = 0
    while offset + len(positions) < len(order) and len(positions) < max_sequences:
        i = int(order[offset + len(positions)])
        updated = max(longest, int(lengths[i]) + 2)
        if positions and updated * (len(positions) + 1) > token_budget:
            break
        positions.append(i)
        longest = updated
    require(positions, "empty extraction batch")
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="check shortest/longest sequence; write nothing")
    args = parser.parse_args()
    root = args.root
    home = Path(__file__).resolve().parent
    config = json.loads((home / "config.json").read_text(encoding="utf-8"))
    require(config["model_id"] == "M4" and config["hidden_size"] == 1024 and
            config["backbone"] == "Rostlab/prot_bert", "invalid M4 config")
    frames = load_development(root)
    frame = __import__("pandas").concat([frames["train"], frames["validation"]], ignore_index=True)
    total = len(frame)
    require(total == 13652, "wrong train+validation total")
    sequences = frame["sequence"].tolist()
    lengths = frame["length"].to_numpy(dtype=np.int64)
    digest = ids_digest(frame)
    config_sha = sha_bytes(canonical_json(config))
    tokenizer, model, commit = load_encoder(config)
    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
        for label, index in (("shortest", int(np.argmin(lengths))),
                             ("longest", int(np.argmax(lengths)))):
            x = encode_batch([sequences[index]], tokenizer, model)
            require(x.shape == (1, 1024) and x.dtype == np.float16, "smoke embedding shape/dtype")
            print(f"SMOKE PASS: {label} ({int(lengths[index])} residues), shape={x.shape}", flush=True)
        print("Peak GPU VRAM GiB:", round(torch.cuda.max_memory_allocated()/2**30, 2))
        print("Encoder smoke completed; NO project files written.")
        return

    out = home / "embeddings"
    require_no_partial(out)
    out.mkdir(parents=True, exist_ok=True)
    npy = out / "pooled_fp16_train_validation.npy"
    meta_path = out / "pooled_fp16_train_validation.json"
    progress_path = out / "extract_progress.json"
    metadata = {
        "schema": "amr_v2_m4_protbert_embeddings_v1",
        "status": "incomplete",
        "backbone": config["backbone"],
        "backbone_commit": commit,
        "backbone_revision_requested": config["backbone_revision"],
        "pooling": config["pooling"],
        "dtype": "float16",
        "shape": [total, int(config["hidden_size"])],
        "partition_order": ["train", "validation"],
        "partition_rows": {name: len(frames[name]) for name in ("train", "validation")},
        "partition_ids_sha256": {name: ids_digest(frames[name]) for name in ("train", "validation")},
        "all_ids_sha256": digest,
        "frozen_train_validation_sha256": SPLIT_HASHES,
        "config_sha256": config_sha,
        "audit_report_sha256": "f587a237fdf4a48b1beaa559cdd56e2461cf46b600b1038e3c5969eb29a6e8ac",
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        for k, expected in metadata.items():
            if k == "status":
                continue
            require(existing.get(k) == expected, f"embedding cache metadata mismatch: {k}")
        require(npy.is_file(), "embedding metadata present but .npy missing")
        if existing["status"] == "complete":
            require(existing.get("file_sha256") == sha256(npy), "completed embedding cache SHA256 mismatch")
            print("Verified completed embedding cache:", npy)
            return
        require(existing["status"] == "incomplete", "unknown embedding cache state")
        require(progress_path.is_file(), "incomplete embedding cache lacks progress cursor")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        require(progress["all_ids_sha256"] == digest and progress["config_sha256"] == config_sha,
                "embedding cursor refers to different data/config")
        cursor = int(progress["next_offset"])
        require(0 <= cursor <= total, "embedding progress cursor out of range")
        array = np.load(npy, mmap_mode="r+")
        require(array.shape == (total, 1024) and array.dtype == np.float16, "existing embedding file shape/dtype")
        print(f"Resuming embedding extraction at {cursor}/{total}", flush=True)
    else:
        require(not npy.exists() and not progress_path.exists(),
                "orphaned embedding file or progress cursor; inspect manually")
        array = np.lib.format.open_memmap(npy, mode="w+", shape=(total, 1024), dtype=np.float16)
        array.flush()
        atomic_json(meta_path, metadata)
        cursor = 0
        atomic_json(progress_path, {
            "next_offset": cursor, "all_ids_sha256": digest, "config_sha256": config_sha})

    # A fixed length-based order reduces padding, but every row is written
    # back to its original train/validation position.
    order = np.argsort(lengths, kind="stable")
    max_batch = int(config["embedding_max_sequences_per_batch"])
    budget = int(config["embedding_max_padded_tokens_per_batch"])
    checkpoint_every = int(config["embedding_progress_every_n_sequences"])
    last = cursor
    t0 = time.time()
    while cursor < total:
        indexes = plan_batch(order, cursor, lengths, max_batch, budget)
        encoded = encode_batch([sequences[i] for i in indexes], tokenizer, model)
        require(encoded.shape == (len(indexes), 1024), "wrong encoder output shape")
        array[indexes] = encoded
        cursor += len(indexes)
        if cursor - last >= checkpoint_every or cursor == total:
            array.flush()
            atomic_json(progress_path, {
                "next_offset": cursor, "all_ids_sha256": digest, "config_sha256": config_sha})
            print(f"Embedded {cursor:,}/{total:,}; elapsed {time.time()-t0:.0f}s; "
                  f"peak GPU allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
            last = cursor
    array.flush()
    del array
    final_hash = sha256(npy)
    atomic_json(meta_path, {**metadata, "status": "complete", "file_sha256": final_hash})
    progress_path.unlink()
    print("M4 EMBEDDINGS COMPLETE", npy, "SHA256:", final_hash)

if __name__ == "__main__":
    main()
