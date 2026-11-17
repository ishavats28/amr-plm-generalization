#!/usr/bin/env python3
"""Train the frozen ProtBERT BiLSTM–Transformer head using training and validation data only.

--smoke: synthetic forward/backprop + padding invariance; writes no project files.
Full training: verifies independent audit, M5 cache and provenance, stages local
cache, trains seed 42, checkpoint selection by validation BCE loss only.
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, roc_auc_score, matthews_corrcoef,
                             precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (atomic_json, canonical_json, require, require_no_partial,
                    sha256, sha_bytes)
from cache import inspect_cache, stage_cache
from model import ResidueBiLSTMTransformer

COLUMNS = ['epoch', 'train_loss', 'validation_loss', 'validation_mcc_0_5',
           'validation_f1_0_5', 'validation_average_precision', 'validation_roc_auc',
           'validation_balanced_accuracy_0_5', 'epoch_seconds', 'best_epoch_so_far',
           'epochs_without_improvement']


def create_model(cfg):
    return ResidueBiLSTMTransformer(
        input_dim=int(cfg['hidden_size']), model_dim=int(cfg['transformer_model_dim']),
        heads=int(cfg['transformer_heads']), transformer_layers=int(cfg['transformer_layers']),
        lstm_hidden_per_direction=int(cfg['lstm_hidden_per_direction']),
        lstm_layers=int(cfg['lstm_layers']),
        lstm_output_dropout=float(cfg['lstm_output_dropout']),
        feedforward_dim=int(cfg['transformer_feedforward_dim']),
        head_widths=tuple(cfg['head_widths']), dropout=float(cfg['dropout']),
        transformer_dropout=float(cfg['transformer_dropout']))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_batches(indices, lengths, max_items, budget, *, shuffle, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        indices = indices[torch.randperm(len(indices), generator=rng).numpy()]
    result = []
    block_size = 256 if shuffle else len(indices)
    for start in range(0, len(indices), block_size):
        block = indices[start:start + block_size]
        block = block[np.argsort(lengths[block], kind='stable')]
        cursor = 0
        while cursor < len(block):
            end, max_len = cursor, 0
            while end < len(block) and end-cursor < max_items:
                proposed = max(max_len, int(lengths[block[end]]))
                if end > cursor and proposed * (end-cursor+1) > budget:
                    break
                end += 1
                max_len = proposed
            require(end > cursor, 'empty M7 batch')
            require(max_len * (end-cursor) <= budget, 'single item exceeds M7 padded budget')
            result.append(block[cursor:end])
            cursor = end
    if shuffle:
        order = torch.randperm(len(result), generator=rng).tolist()
        result = [result[j] for j in order]
    require(sum(len(part) for part in result) == len(indices), 'batch coverage failure')
    require(len(set(int(i) for b in result for i in b)) == len(indices), 'duplicate indices in batches')
    return result


def collate(cache, offsets, lengths, labels, indices, device):
    max_len = int(lengths[indices].max())
    features = np.zeros((len(indices), max_len, cache.shape[1]), dtype=np.float32)
    for row, idx in enumerate(indices):
        start, stop, length = int(offsets[idx]), int(offsets[idx+1]), int(lengths[idx])
        require(stop - start == length, 'ragged cache row length mismatch')
        features[row, :length] = cache[start:stop]
    return (torch.from_numpy(features).to(device),
            torch.from_numpy(np.asarray(lengths[indices], dtype=np.int64)).to(device),
            torch.from_numpy(np.asarray(labels[indices], dtype=np.float32)).to(device))


def evaluate(model, cache, offsets, lengths, labels, batches, device):
    model.eval()
    summed = nn.BCEWithLogitsLoss(reduction='sum')
    total, probs, truths = 0.0, [], []
    with torch.inference_mode():
        for indices in batches:
            x, size, y = collate(cache, offsets, lengths, labels, indices, device)
            logits = model(x, size)
            total += float(summed(logits, y).item())
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            truths.append(y.detach().cpu().numpy())
    y = np.concatenate(truths).astype(np.int64)
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(np.int64)
    require(np.isfinite(p).all(), 'nonfinite M7 validation probabilities')
    return {
        'loss': total / len(y),
        'mcc': float(matthews_corrcoef(y, pred)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
    }


def atomic_checkpoint(path, obj):
    staging = path.with_name(path.name+'.partial')
    require(not staging.exists(), f'stale checkpoint stage {staging}')
    torch.save(obj, staging)
    os.replace(staging, path)


def write_history(path, rows):
    staging = path.with_name(path.name+'.partial')
    require(not staging.exists(), f'stale history stage {staging}')
    with staging.open('x', newline='', encoding='utf-8') as handle:
        out = csv.DictWriter(handle, fieldnames=COLUMNS)
        out.writeheader()
        out.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def smoke(cfg):
    """No real data or output files. Test padding masks, recurrent gradients and long GPU batches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(int(cfg['seed']))
    model = create_model(cfg).to(device)
    lengths = torch.tensor([35, 47, 55], dtype=torch.long, device=device)
    x = torch.randn(3, 55, int(cfg['hidden_size']), device=device)
    valid = torch.arange(55, device=device)[None, :] < lengths[:, None]
    corrupted = x.masked_fill(~valid.unsqueeze(-1), 1234.)
    extended = torch.cat([corrupted, torch.randn(3, 23, int(cfg['hidden_size']), device=device) * 100], dim=1)
    model.eval()
    with torch.inference_mode():
        a = model(x, lengths)
        b = model(extended, lengths)
        require(a.shape == (3,) and torch.allclose(a, b, atol=3e-4, rtol=3e-4),
                'masked padding changes M7 predictions')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']),
                                  weight_decay=float(cfg['weight_decay']))
    targets = torch.tensor([0., 1., 0.], device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.BCEWithLogitsLoss()(model(x, lengths), targets)
    require(bool(torch.isfinite(loss)), 'nonfinite M7 synthetic loss')
    loss.backward()
    lstm = model.bilstm.weight_ih_l0.grad
    reverse = model.bilstm.weight_ih_l0_reverse.grad
    attention = model.encoder.layers[0].self_attn.in_proj_weight.grad
    for label, grad in [('BiLSTM', lstm), ('reverse BiLSTM', reverse), ('attention', attention)]:
        require(grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0,
                f'M7 missing/invalid {label} gradient')
    nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip_norm']))
    optimizer.step()
    print('M7 SYNTHETIC HEAD SMOKE PASS: masking, packed LSTM forward/reverse, attention, optimizer', flush=True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad(set_to_none=True)
        # Covers the validated 1014-residue maximum with four highly padded examples.
        # Any CUDA OOM stops before the full training run. No auto-batch mutation.
        long_x = torch.randn(4, 1014, int(cfg['hidden_size']), device=device)
        long_lengths = torch.tensor([1014, 1014, 900, 800], device=device)
        long_loss = nn.BCEWithLogitsLoss()(model(long_x, long_lengths),
                                           torch.tensor([1., 0., 1., 0.], device=device))
        require(bool(torch.isfinite(long_loss)), 'M7 long-sequence synthetic loss nonfinite')
        long_loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f'M7 GPU LONG-LENGTH SMOKE PASS: 4 x 1014 padded | allocated peak {peak:.2f} GiB', flush=True)
    print('No dataset or project files written. Test/OOD not accessed.', flush=True)


def train(root, cfg, home):
    require(torch.cuda.is_available(), 'M7 real training requires CUDA')
    require(cfg['model_id'] == 'M7' and cfg['backbone'] == 'Rostlab/prot_bert' and cfg['encoder_frozen'] is True and
            cfg['checkpoint_metric'] == 'validation_loss' and cfg['head_precision'] == 'float32',
            'M7 scientific configuration mismatch')
    require(cfg['backbone_revision'] == '7a894481acdc12202f0a415dd567f6cfdb698908',
            'ProtBERT backbone revision changed')
    seed = int(cfg['seed'])
    set_seed(seed)
    device = torch.device('cuda')
    print('M7 GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Cache reused from M5; no ProtBERT re-encoding; test/OOD NEVER opened', flush=True)
    # Enforce the frozen-model training order and prior immutable checkpoint.
    predecessor = root / 'training_v2/M6_protbert_transformer_attention/runs/seed42'
    prior_summary_file = predecessor / 'training_summary.json'
    require(prior_summary_file.is_file(), 'M6 must finish before M7')
    prior = json.loads(prior_summary_file.read_text(encoding='utf-8'))
    require(prior.get('status') == 'complete' and prior.get('dataset_version') == 'V2'
            and prior.get('model_id') == 'M6' and prior.get('seed') == seed
            and prior.get('checkpoint_metric') == 'validation_loss'
            and prior.get('selection_uses_test_or_ood') is False,
            'M6 predecessor completion or provenance mismatch')
    require(prior.get('best_model_sha256') == '6b5ee31527c5543be1df0bfaa2fe9642b114c1365af68734018affffea214c0c',
            'M6 independently replayed checkpoint digest mismatch')
    require((predecessor / 'best_model.pt').is_file() and
            sha256(predecessor / 'best_model.pt') == prior.get('best_model_sha256'),
            'M6 selected checkpoint changed')
    frame, train_count, lengths, offsets, meta, cache_path = inspect_cache(root, cfg, hash_full=True)
    require(prior.get('embedding_sha256') == meta['file_sha256'],
            'M6 and M7 residue-cache provenance differs')
    local_path = stage_cache(cache_path, meta['file_sha256'])
    arr = np.load(local_path, mmap_mode='r', allow_pickle=False)
    require(arr.shape == (4229962,1024) and arr.dtype == np.float16, 'staged M5 cache shape/type mismatch')
    # Cache verification is complete before creating run artifacts.
    labels = frame['label'].to_numpy(dtype=np.int64)
    run = home/'runs'/f'seed{seed}'
    require_no_partial(run)
    summary_file, history_file = run/'training_summary.json', run/'history.csv'
    best_file, last_file, config_file = run/'best_model.pt', run/'last_checkpoint.pt', run/'config.json'
    require(not summary_file.exists(), 'M7 already completed; refusing overwrite')
    run.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        require(json.loads(config_file.read_text()) == cfg, 'M7 saved run config differs')
    else:
        require(not any(p.exists() for p in (best_file,last_file,history_file)),
                'M7 orphaned checkpoints/history without run config')
        atomic_json(config_file, cfg)
    model = create_model(cfg).to(device)
    print('M7 trainable parameters:', sum(x.numel() for x in model.parameters()), flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']),
                               weight_decay=float(cfg['weight_decay']))
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    max_items = int(cfg['batch_size'])
    budget = int(cfg['max_padded_residues_per_batch'])
    validation_batches = make_batches(np.arange(train_count,len(frame)), lengths, max_items,
                                      budget, shuffle=False, rng=generator)
    best_epoch, no_improvement, best_loss, first = 0, 0, float('inf'), 1
    history = []
    conf_sha = sha_bytes(canonical_json(cfg))
    if last_file.exists():
        saved = torch.load(last_file,map_location='cpu',weights_only=True)
        require(saved['config_sha256'] == conf_sha and saved['embedding_sha256'] == meta['file_sha256'],
                'M7 resume checkpoint provenance mismatch')
        require(best_file.is_file(), 'M7 cannot resume without best checkpoint')
        model.load_state_dict(saved['model'])
        optim.load_state_dict(saved['optimizer'])
        for state in optim.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        first = int(saved['epoch'])+1
        best_epoch = int(saved['best_epoch'])
        best_loss = float(saved['best_loss'])
        no_improvement = int(saved['patience_count'])
        history = list(saved['history'])
        require(len(history) == first-1, 'resume history length mismatch')
        generator.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print('M7 resuming from epoch', first, flush=True)
    else:
        require(not best_file.exists() and not history_file.exists(), 'M7 orphaned run state')
    start = time.monotonic()
    stop_reason = 'max_epochs'
    for epoch in range(first, int(cfg['max_epochs'])+1):
        if no_improvement >= int(cfg['early_stopping_patience']):
            stop_reason = 'early_stopping'
            break
        tic = time.monotonic()
        model.train()
        train_batches = make_batches(np.arange(train_count), lengths, max_items, budget,
                                     shuffle=True, rng=generator)
        total_loss = 0.0
        for indices in train_batches:
            x, sizes, targets = collate(arr, offsets, lengths, labels, indices, device)
            optim.zero_grad(set_to_none=True)
            logits = model(x, sizes)
            loss = criterion(logits, targets)
            require(bool(torch.isfinite(loss)), 'nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip_norm']))
            optim.step()
            total_loss += float(loss.detach().item())*len(indices)
        result = evaluate(model, arr, offsets, lengths, labels, validation_batches, device)
        require(np.isfinite(result['loss']), 'nonfinite validation BCE')
        improved = result['loss'] < best_loss - float(cfg['early_stopping_min_delta'])
        if improved:
            best_epoch = epoch
            best_loss = float(result['loss'])
            no_improvement = 0
            atomic_checkpoint(best_file, {
                'epoch': epoch, 'model': model.state_dict(), 'config_sha256': conf_sha,
                'embedding_sha256': meta['file_sha256'], 'backbone_commit': cfg['backbone_revision'],
                'selection_criterion': 'validation_loss',
            })
        else:
            no_improvement += 1
        history.append({
            'epoch': epoch, 'train_loss': total_loss/train_count,
            'validation_loss': result['loss'], 'validation_mcc_0_5': result['mcc'],
            'validation_f1_0_5': result['f1'], 'validation_average_precision': result['average_precision'],
            'validation_roc_auc': result['roc_auc'], 'validation_balanced_accuracy_0_5': result['balanced_accuracy'],
            'epoch_seconds': time.monotonic()-tic,
            'best_epoch_so_far': best_epoch, 'epochs_without_improvement': no_improvement,
        })
        atomic_checkpoint(last_file, {
            'epoch': epoch, 'model': model.state_dict(), 'optimizer': optim.state_dict(),
            'config_sha256': conf_sha, 'embedding_sha256': meta['file_sha256'],
            'backbone_commit': cfg['backbone_revision'],
            'best_loss': best_loss, 'best_epoch': best_epoch, 'patience_count': no_improvement,
            'history': history, 'sampler_rng': generator.get_state(), 'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),
        })
        write_history(history_file, history)
        print(f'Epoch {epoch:02d}/{int(cfg["max_epochs"])}: '
              f'train_loss={total_loss/train_count:.5f} val_loss={result["loss"]:.5f} '
              f'val_MCC={result["mcc"]:.4f} best={best_epoch} unimproved={no_improvement}',flush=True)
        if no_improvement >= int(cfg['early_stopping_patience']):
            stop_reason = 'early_stopping'
            break
    require(best_file.is_file() and last_file.is_file() and history_file.is_file() and best_epoch>0,
            'M7 training lacks required checkpoints')
    selected = next(r for r in history if r['epoch']==best_epoch)
    require(abs(float(selected['validation_loss'])-best_loss)<1e-9,
            'M7 selected checkpoint/epoch inconsistency')
    if no_improvement >= int(cfg['early_stopping_patience']):
        stop_reason = 'early_stopping'
    atomic_json(summary_file, {
        'status':'complete','dataset_version':'V2','model_id':'M7','seed':seed,
        'train_rows':train_count,'validation_rows':len(frame)-train_count,
        'checkpoint_metric':'validation_loss','selection_uses_test_or_ood':False,
        'best_epoch':best_epoch,'epochs_executed':len(history), 'best_validation_loss':best_loss,
        'best_validation_metrics_at_0_5': {
            'mcc':selected['validation_mcc_0_5'],'f1':selected['validation_f1_0_5'],
            'roc_auc':selected['validation_roc_auc'],
            'average_precision':selected['validation_average_precision'],
            'balanced_accuracy':selected['validation_balanced_accuracy_0_5'],
        },
        'stop_reason':stop_reason,'elapsed_seconds':time.monotonic()-start,
        'config_sha256':conf_sha,'embedding_sha256':meta['file_sha256'],
        'embedding_source':str(cache_path.relative_to(root)),
        'backbone_commit':cfg['backbone_revision'],
        'audit_report_sha256':meta['audit_report_sha256'],
        'best_model_sha256':sha256(best_file),'last_checkpoint_sha256':sha256(last_file),
        'history_sha256':sha256(history_file),
        'torch_version':torch.__version__, 'gpu':torch.cuda.get_device_name(0),
    })
    print('M7 TRAINING COMPLETE; best epoch:',best_epoch,flush=True)
    print('Summary:',summary_file,flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    home = Path(__file__).resolve().parent
    cfg = json.loads((home/'config.json').read_text(encoding='utf-8'))
    require(cfg.get('model_id')=='M7' and cfg.get('checkpoint_metric')=='validation_loss',
            'M7 config invalid')
    if args.smoke:
        smoke(cfg)
    else:
        train(args.root, cfg, home)

if __name__=='__main__':
    main()
