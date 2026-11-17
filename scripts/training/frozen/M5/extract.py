#!/usr/bin/env python3
"""Extract frozen ProtBERT residue embeddings for M5 using training and validation data only.

--smoke: GPU encoder parity check with M4 pooled embeddings; writes nothing.
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
from transformers import BertModel, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (SPLIT_HASHES, atomic_json, canonical_json, ids_digest,
                    load_development, require, require_no_partial, sha256, sha_bytes)


def load_encoder(config):
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'BF16 CUDA GPU required for M5 ProtBERT extraction')
    revision = config['backbone_revision']
    require(isinstance(revision, str) and len(revision) == 40 and
            all(c in '0123456789abcdef' for c in revision), 'M5 requires immutable ProtBERT commit')
    name = config['backbone']
    require(name == 'Rostlab/prot_bert', 'wrong ProtBERT backbone')
    print(f'Loading frozen encoder {name} @ {revision} (BF16)', flush=True)
    tokenizer = BertTokenizer.from_pretrained(name, revision=revision, do_lower_case=False)
    model = BertModel.from_pretrained(name, revision=revision,
                                     add_pooling_layer=False, dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to('cuda')
    require(int(model.config.hidden_size) == 1024, 'wrong ProtBERT hidden size')
    return tokenizer, model


def prep_sequence(sequence):
    sequence = str(sequence).upper()
    sequence = sequence.translate(str.maketrans({'U': 'X', 'Z': 'X', 'O': 'X', 'B': 'X'}))
    require(set(sequence) <= set('ACDEFGHIKLMNPQRSTVWYX'), 'unexpected amino-acid alphabet')
    return ' '.join(sequence)


def encode_residues(sequences, tokenizer, model):
    # Same tokenization, BF16 backbone, and residue mask as M4 pooled extraction.
    payload = tokenizer([prep_sequence(s) for s in sequences],
                        return_tensors='pt', padding=True, truncation=False,
                        return_special_tokens_mask=True)
    ids = payload['input_ids'].to('cuda')
    attention = payload['attention_mask'].to('cuda')
    special = payload['special_tokens_mask'].to('cuda')
    require(ids.shape[1] <= int(model.config.max_position_embeddings),
            'ProtBERT maximum positional length exceeded')
    mask = attention.bool() & ~special.bool()
    require(mask.sum(dim=1).cpu().tolist() == [len(s) for s in sequences],
            'ProtBERT tokenizer has lost or introduced residues')
    require(not bool((ids[mask] == tokenizer.unk_token_id).any().item()),
            'ProtBERT returned unknown residue token')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = model(input_ids=ids, attention_mask=attention).last_hidden_state
        outputs = [hidden[i, mask[i]].float().cpu().numpy().astype(np.float16)
                   for i in range(len(sequences))]
    require(all(x.shape == (len(s), 1024) and np.isfinite(x).all()
                for x, s in zip(outputs, sequences)), 'invalid ProtBERT residue features')
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
    require(config['model_id'] == 'M5' and config['hidden_size'] == 1024, 'wrong M5 configuration')
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
    m4home = args.root / 'training_v2/M4_protbert_mlp'
    m4meta = json.loads((m4home / 'embeddings/pooled_fp16_train_validation.json').read_text())
    require(m4meta['schema'] == 'amr_v2_m4_protbert_embeddings_v1' and m4meta['status'] == 'complete' and m4meta['backbone_commit'] == config['backbone_revision'],
            'M4 model revision mismatch')
    require(m4meta['all_ids_sha256'] == all_ids_hash and
            m4meta['frozen_train_validation_sha256'] == SPLIT_HASHES,
            'M4 dataset or protein ordering differs from the certified partitions')
    m4conf = json.loads((m4home / 'config.json').read_text())
    require(m4conf['backbone_revision'] == config['backbone_revision'], 'M4 config backbone changed')
    require(m4meta['config_sha256'] == sha_bytes(canonical_json(m4conf)), 'M4 metadata/config checksum mismatch')
    m4vectors = m4home / 'embeddings/pooled_fp16_train_validation.npy'
    require(sha256(m4vectors) == m4meta['file_sha256'], 'M4 cache checksum mismatch')
    tokenizer, model = load_encoder(config)
    if args.smoke:
        m4 = np.load(m4vectors, mmap_mode='r', allow_pickle=False)
        require(m4.shape == (len(frame), 1024), 'M4 pooled shape mismatch')
        for name, i in (('shortest', int(lengths.argmin())), ('longest', int(lengths.argmax()))):
            result = encode_residues([sequences[i]], tokenizer, model)[0]
            pooled = result.astype(np.float32).mean(axis=0)
            original = np.asarray(m4[i], dtype=np.float32)
            max_error = float(np.max(np.abs(pooled - original)))
            require(np.allclose(pooled, original, atol=0.02, rtol=0.02),
                    f'{name}: M5 residue embedding mean differs from M4 checkpoint output; max error {max_error}')
            print(f'M5 SMOKE PASS: {name}, length={lengths[i]}, shape={result.shape}, '
                  f'max_mean_difference_from_M4={max_error:.6g}', flush=True)
        print('M5 SMOKE COMPLETE; no output files written; test/OOD not read.')
        return

    outdir = home / 'embeddings'
    require_no_partial(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npy = outdir / 'residue_fp16_train_validation.npy'
    offset_path = outdir / 'residue_offsets.npy'
    meta_path = outdir / 'residue_fp16_train_validation.json'
    cursor_path = outdir / 'extract_progress.json'
    expected_meta = {
        'schema': 'amr_v2_m5_protbert_residue_embeddings_v1',
        'status': 'incomplete',
        'backbone': config['backbone'],
        'backbone_commit': config['backbone_revision'],
        'dtype': 'float16',
        'shape': [count_residues, 1024],
        'partition_order': ['train', 'validation'],
        'partition_rows': {key: len(frames[key]) for key in ('train', 'validation')},
        'partition_ids_sha256': {key: ids_digest(frames[key]) for key in ('train', 'validation')},
        'all_ids_sha256': all_ids_hash,
        'frozen_train_validation_sha256': SPLIT_HASHES,
        'audit_report_sha256': m4meta['audit_report_sha256'],
        'm4_pooled_file_sha256': m4meta['file_sha256'],
        'config_sha256': config_sha,
        'offsets_values_sha256': hashlib.sha256(offsets.tobytes()).hexdigest(),
    }
    require(count_residues > 0, 'empty cache')
    print(f'Building ragged M5 residue cache: {len(frame):,} sequences, {count_residues:,} residues, '
          f'{(count_residues*2048+128)/2**30:.2f} GiB', flush=True)
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
            print('Verified existing complete M5 residue cache; no changes written.')
            return
        require(existing['status'] == 'incomplete' and cursor_path.is_file(), 'incomplete cache lacks cursor')
        progress = json.loads(cursor_path.read_text())
        require(progress.get('all_ids_sha256') == all_ids_hash and
                progress.get('config_sha256') == config_sha, 'cursor belongs to another experiment')
        cursor = int(progress['next_offset'])
        require(0 <= cursor <= len(frame), 'invalid cursor')
        data = np.load(npy, mmap_mode='r+', allow_pickle=False)
        require(data.shape == (count_residues, 1024) and data.dtype == np.float16, 'cache shape invalid')
        print(f'Resuming at sequence {cursor}/{len(frame)}', flush=True)
    else:
        require(not any(path.exists() for path in (npy, offset_path, cursor_path)),
                'orphaned cache/offset/cursor; inspect, do not overwrite')
        print('Creating large Drive cache. Interruptions will resume from flushed cursor.', flush=True)
        data = np.lib.format.open_memmap(npy, mode='w+', dtype=np.float16,
                                         shape=(count_residues, 1024))
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
            print(f'M5 embedded {cursor:,}/{len(frame):,}; elapsed {time.monotonic()-tic:.0f}s', flush=True)
    data.flush()
    del data
    result_sha = sha256(npy)
    atomic_json(meta_path, {**expected_meta, 'status': 'complete',
                            'offsets_file_sha256': sha256(offset_path),
                            'file_sha256': result_sha})
    cursor_path.unlink()
    print('M5 RESIDUE EMBEDDINGS COMPLETE; SHA256:', result_sha, flush=True)


if __name__ == '__main__':
    main()
