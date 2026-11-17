#!/usr/bin/env python3
"""Extract frozen ESM-2 residue embeddings for M1 using training and validation data only.

--smoke: GPU encoder parity check with M0 pooled embeddings; writes nothing.
--extract: resumable ragged FP16 cache on Drive (large; run preflight first).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (SPLIT_HASHES, atomic_json, canonical_json, ids_digest,
                    load_development, require, require_no_partial, sha256, sha_bytes)


def load_encoder(config):
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'BF16 CUDA GPU required for M1 ESM-2 extraction')
    revision = config['backbone_revision']
    require(isinstance(revision, str) and len(revision) == 40 and
            all(c in '0123456789abcdef' for c in revision), 'M1 needs immutable 40-char ESM-2 revision')
    name = config['backbone']
    print(f'Loading frozen encoder {name} @ {revision} (BF16)', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
    model = AutoModel.from_pretrained(name, revision=revision, dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to('cuda')
    require(model.config.hidden_size == 1280, 'unexpected pretrained hidden size')
    return tokenizer, model


def encode_residues(sequences, tokenizer, model):
    payload = tokenizer(sequences, padding=True, truncation=False,
                        return_special_tokens_mask=True, return_tensors='pt')
    token_ids = payload['input_ids'].to('cuda')
    attention = payload['attention_mask'].to('cuda')
    special = payload['special_tokens_mask'].to('cuda')
    require(token_ids.shape[1] <= 1024, 'ESM-2 maximum token length exceeded')
    residue = attention.bool() & ~special.bool()
    observed = residue.sum(1).cpu().tolist()
    require(observed == [len(seq) for seq in sequences], 'tokenizer residue lengths differ')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = model(input_ids=token_ids, attention_mask=attention).last_hidden_state
        outputs = [hidden[i, residue[i]].to(torch.float32).cpu().numpy().astype(np.float16)
                   for i in range(len(sequences))]
    require(all(x.shape == (len(seq), 1280) and np.isfinite(x).all()
                for x, seq in zip(outputs, sequences)), 'invalid residue embedding')
    return outputs


def plan_batch(order, cursor, lengths, max_items, max_tokens):
    indices = []
    longest = 0
    while cursor + len(indices) < len(order) and len(indices) < max_items:
        idx = int(order[cursor + len(indices)])
        proposed_longest = max(longest, int(lengths[idx]) + 2)
        if indices and proposed_longest * (len(indices) + 1) > max_tokens:
            break
        indices.append(idx)
        longest = proposed_longest
    require(indices, 'empty encoder batch')
    return indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--extract', action='store_true')
    args = ap.parse_args()
    home = Path(__file__).resolve().parent
    config = json.loads((home / 'config.json').read_text())
    require(config['model_id'] == 'M1' and config['hidden_size'] == 1280, 'wrong M1 configuration')
    frames = load_development(args.root)
    frame = __import__('pandas').concat([frames['train'], frames['validation']], ignore_index=True)
    require(len(frame) == 13652, 'incorrect training and validation count')
    lengths = frame['length'].to_numpy(dtype=np.int64)
    sequences = frame['sequence'].tolist()
    offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    count_residues = int(offsets[-1])
    all_ids_hash = ids_digest(frame)
    config_sha = sha_bytes(canonical_json(config))
    m0home = args.root / 'training_v2/M0_esm2_mlp'
    m0meta = json.loads((m0home / 'embeddings/pooled_fp16_train_validation.json').read_text())
    require(m0meta['status'] == 'complete' and m0meta['backbone_commit'] == config['backbone_revision'],
            'M0 model revision mismatch')
    require(m0meta['all_ids_sha256'] == all_ids_hash and
            m0meta['frozen_train_validation_sha256'] == SPLIT_HASHES,
            'M0 dataset or protein ordering differs from the certified partitions')
    m0vectors = m0home / 'embeddings/pooled_fp16_train_validation.npy'
    require(sha256(m0vectors) == m0meta['file_sha256'], 'M0 cache checksum mismatch')
    tokenizer, model = load_encoder(config)
    if args.smoke:
        m0 = np.load(m0vectors, mmap_mode='r', allow_pickle=False)
        require(m0.shape == (len(frame), 1280), 'M0 pooled shape mismatch')
        for name, i in (('shortest', int(lengths.argmin())), ('longest', int(lengths.argmax()))):
            result = encode_residues([sequences[i]], tokenizer, model)[0]
            pooled = result.astype(np.float32).mean(axis=0)
            original = np.asarray(m0[i], dtype=np.float32)
            max_error = float(np.max(np.abs(pooled - original)))
            require(np.allclose(pooled, original, atol=0.01, rtol=0.01),
                    f'{name}: M1 residue embedding mean differs from M0 checkpoint output; max error {max_error}')
            print(f'M1 SMOKE PASS: {name}, length={lengths[i]}, shape={result.shape}, '
                  f'max_mean_difference_from_M0={max_error:.6g}', flush=True)
        print('M1 SMOKE COMPLETE; no output files written; test/OOD not read.')
        return

    outdir = home / 'embeddings'
    require_no_partial(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npy = outdir / 'residue_fp16_train_validation.npy'
    offset_path = outdir / 'residue_offsets.npy'
    meta_path = outdir / 'residue_fp16_train_validation.json'
    cursor_path = outdir / 'extract_progress.json'
    expected_meta = {
        'schema': 'amr_v2_m1_residue_embeddings_v1',
        'status': 'incomplete',
        'backbone': config['backbone'],
        'backbone_commit': config['backbone_revision'],
        'dtype': 'float16',
        'shape': [count_residues, 1280],
        'partition_order': ['train', 'validation'],
        'partition_rows': {key: len(frames[key]) for key in ('train', 'validation')},
        'partition_ids_sha256': {key: ids_digest(frames[key]) for key in ('train', 'validation')},
        'all_ids_sha256': all_ids_hash,
        'frozen_train_validation_sha256': SPLIT_HASHES,
        'audit_report_sha256': m0meta['audit_report_sha256'],
        'm0_pooled_file_sha256': m0meta['file_sha256'],
        'config_sha256': config_sha,
        'offsets_values_sha256': hashlib.sha256(offsets.tobytes()).hexdigest(),
    }
    require(count_residues > 0, 'empty cache')
    print(f'Building ragged M1 residue cache: {len(frame):,} sequences, {count_residues:,} residues, '
          f'{(count_residues*2560+128)/2**30:.2f} GiB', flush=True)
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        for key, value in expected_meta.items():
            if key != 'status':
                require(existing.get(key) == value, f'cache metadata mismatch: {key}')
        require(npy.is_file() and offset_path.is_file(), 'cache companion missing')
        require(existing.get('offsets_file_sha256') == sha256(offset_path), 'offsets file checksum changed')
        old_offsets = np.load(offset_path, allow_pickle=False)
        require(np.array_equal(old_offsets, offsets), 'cache offsets differ from certified protein data')
        if existing['status'] == 'complete':
            require(sha256(npy) == existing.get('file_sha256'), 'completed cache SHA mismatch')
            require(not cursor_path.exists(), 'completed cache has stale progress cursor')
            print('Verified existing complete M1 residue cache; no changes written.')
            return
        require(existing['status'] == 'incomplete' and cursor_path.is_file(), 'incomplete cache lacks cursor')
        progress = json.loads(cursor_path.read_text())
        require(progress.get('all_ids_sha256') == all_ids_hash and
                progress.get('config_sha256') == config_sha, 'cursor belongs to another experiment')
        cursor = int(progress['next_offset'])
        require(0 <= cursor <= len(frame), 'invalid cursor')
        data = np.load(npy, mmap_mode='r+', allow_pickle=False)
        require(data.shape == (count_residues, 1280) and data.dtype == np.float16, 'cache shape invalid')
        print(f'Resuming at sequence {cursor}/{len(frame)}', flush=True)
    else:
        require(not any(path.exists() for path in (npy, offset_path, cursor_path)),
                'orphaned cache/offset/cursor; inspect, do not overwrite')
        print('Creating large Drive cache. Interruptions will resume from flushed cursor.', flush=True)
        data = np.lib.format.open_memmap(npy, mode='w+', dtype=np.float16,
                                         shape=(count_residues, 1280))
        data.flush()
        # Use np.save once (small file). Never overwrite once metadata exists.
        with offset_path.open('xb') as fh:
            np.save(fh, offsets, allow_pickle=False)
        atomic_json(meta_path, {**expected_meta, 'offsets_file_sha256': sha256(offset_path)})
        cursor = 0
        atomic_json(cursor_path, {'next_offset': cursor, 'config_sha256': config_sha,
                                  'all_ids_sha256': all_ids_hash})
    order = np.argsort(lengths, kind='stable')
    last_saved = cursor
    tic = time.monotonic()
    while cursor < len(frame):
        batch = plan_batch(order, cursor, lengths,
                           int(config['embedding_max_sequences_per_batch']),
                           int(config['embedding_max_padded_tokens_per_batch']))
        vectors = encode_residues([sequences[idx] for idx in batch], tokenizer, model)
        for idx, vector in zip(batch, vectors):
            data[int(offsets[idx]):int(offsets[idx+1])] = vector
        cursor += len(batch)
        if cursor - last_saved >= int(config['embedding_progress_every_n_sequences']) or cursor == len(frame):
            data.flush()
            atomic_json(cursor_path, {'next_offset': cursor, 'config_sha256': config_sha,
                                      'all_ids_sha256': all_ids_hash})
            last_saved = cursor
            print(f'M1 embedded {cursor:,}/{len(frame):,}; elapsed {time.monotonic()-tic:.0f}s', flush=True)
    data.flush()
    del data
    result_sha = sha256(npy)
    atomic_json(meta_path, {**expected_meta, 'status': 'complete',
                            'offsets_file_sha256': sha256(offset_path),
                            'file_sha256': result_sha})
    cursor_path.unlink()
    print('M1 RESIDUE EMBEDDINGS COMPLETE; SHA256:', result_sha, flush=True)


if __name__ == '__main__':
    main()
