#!/usr/bin/env python3
"""Train the M1 multiscale CNN using frozen residue embeddings from training and validation data.

No test or OOD reads; select checkpoint by predeclared minimum validation BCE.
Full training requires CUDA and local staging capacity. --smoke writes nothing.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import shutil
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
from common import (SPLIT_HASHES, atomic_json, canonical_json, ids_digest,
                    load_development, require, require_no_partial, sha256, sha_bytes)
from model import MultiScaleResidueCNN

COLUMNS = ['epoch', 'train_loss', 'validation_loss', 'validation_mcc_0_5',
           'validation_f1_0_5', 'validation_average_precision', 'validation_roc_auc',
           'validation_balanced_accuracy_0_5', 'epoch_seconds', 'best_epoch_so_far',
           'epochs_without_improvement']


def create_model(cfg):
    return MultiScaleResidueCNN(input_dim=int(cfg['hidden_size']),
                                kernels=tuple(cfg['cnn_kernels']),
                                channels=int(cfg['cnn_channels_per_kernel']),
                                widths=tuple(cfg['head_widths']),
                                dropout=float(cfg['dropout']))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_checkpoint(path, payload):
    stage = path.with_name(path.name + '.partial')
    require(not stage.exists(), f'stale checkpoint partial: {stage}')
    torch.save(payload, stage)
    os.replace(stage, path)


def write_history(path, records):
    stage = path.with_name(path.name + '.partial')
    require(not stage.exists(), f'stale history partial: {stage}')
    with stage.open('x', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(stage, path)


def make_batches(indices, lengths, max_items, token_budget, *, shuffle, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        indices = indices[torch.randperm(len(indices), generator=rng).numpy()]
    result = []
    block_size = 256 if shuffle else len(indices)
    for block_start in range(0, len(indices), block_size):
        block = indices[block_start:block_start + block_size]
        block = block[np.argsort(lengths[block], kind='stable')]
        pos = 0
        while pos < len(block):
            end = pos
            largest = 0
            while end < len(block) and end - pos < max_items:
                updated = max(largest, int(lengths[block[end]]))
                if end > pos and updated * (end - pos + 1) > token_budget:
                    break
                largest = updated
                end += 1
            require(end > pos, 'zero-item batch')
            result.append(block[pos:end])
            pos = end
    if shuffle:
        order = torch.randperm(len(result), generator=rng).tolist()
        result = [result[i] for i in order]
    require(sum(len(b) for b in result) == len(indices), 'batch coverage is incomplete')
    return result


def collate(cache, offsets, lengths, labels, indices, device):
    max_length = int(lengths[indices].max())
    features = np.zeros((len(indices), max_length, cache.shape[1]), dtype=np.float32)
    for row, idx in enumerate(indices):
        offset = int(offsets[idx]); endpoint = int(offsets[idx+1]); n = int(lengths[idx])
        require(endpoint - offset == n, 'ragged offset/length inconsistency')
        features[row, :n] = cache[offset:endpoint]
    x = torch.from_numpy(features).to(device)
    sizes = torch.from_numpy(np.asarray(lengths[indices], dtype=np.int64)).to(device)
    y = torch.from_numpy(np.asarray(labels[indices], dtype=np.float32)).to(device)
    return x, sizes, y


def evaluate(model, cache, offsets, lengths, labels, batches, device):
    model.eval()
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    total, probabilities, truth = 0.0, [], []
    with torch.inference_mode():
        for indices in batches:
            features, sizes, targets = collate(cache, offsets, lengths, labels, indices, device)
            logits = model(features, sizes)
            total += float(loss_func(logits, targets).item())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            truth.append(targets.cpu().numpy())
    y = np.concatenate(truth).astype(np.int64)
    p = np.concatenate(probabilities)
    yhat = (p >= 0.5).astype(np.int64)
    require(len(y) == len(np.concatenate(batches)), 'invalid evaluation input length')
    return {
        'loss': total/len(y),
        'mcc': float(matthews_corrcoef(y, yhat)),
        'f1': float(f1_score(y, yhat, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, yhat)),
        'accuracy': float(accuracy_score(y, yhat)),
        'precision': float(precision_score(y, yhat, zero_division=0)),
        'recall': float(recall_score(y, yhat)),
    }


def stage_cache(source: Path, expected_sha: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / ('m1_' + source.name)
    if dest.exists():
        require(sha256(dest) == expected_sha, 'local staged cache digest mismatch')
        print('Using previously verified local M1 cache', flush=True)
        return dest
    bytes_to_copy = source.stat().st_size
    free = shutil.disk_usage(local_dir).free
    require(free > int(bytes_to_copy * 1.15),
            f'Insufficient local disk to stage M1 cache: need {bytes_to_copy*1.15/2**30:.2f} GiB')
    partial = dest.with_name(dest.name + '.partial')
    require(not partial.exists(), 'stale local stage partial; inspect before retry')
    print(f'Staging {bytes_to_copy/2**30:.2f} GiB from source to local cache; verify SHA...', flush=True)
    with source.open('rb') as src, partial.open('xb') as out:
        while True:
            chunk = src.read(16 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    require(sha256(partial) == expected_sha, 'staged cache SHA mismatch')
    os.replace(partial, dest)
    return dest


def smoke(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(int(cfg['seed']))
    head = create_model(cfg).to(device)
    head.eval()
    x = torch.randn(3, 55, 1280, device=device)
    sizes = torch.tensor([40, 49, 55], dtype=torch.long, device=device)
    with torch.inference_mode():
        first = head(x, sizes)
        padded = torch.cat([x, torch.randn(3, 23, 1280, device=device)], dim=1)
        second = head(padded, sizes)
        noisy = x.clone()
        for i in range(3):
            noisy[i, int(sizes[i]):] = torch.randn_like(noisy[i, int(sizes[i]):]) * 100
        third = head(noisy, sizes)
        require(torch.allclose(first, second, atol=1e-5, rtol=1e-5), 'extra right padding changes logits')
        require(torch.allclose(first, third, atol=1e-5, rtol=1e-5), 'masked padding changes logits')
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg['learning_rate']))
    y = torch.tensor([0., 1., 0.], device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.BCEWithLogitsLoss()(head(x, sizes), y)
    require(bool(torch.isfinite(loss)), 'nonfinite synthetic loss')
    loss.backward()
    optimizer.step()
    print('M1 SYNTHETIC CNN SMOKE PASS: masked padding, gradients, optimizer; no files written.')


def train(root, config, home):
    require(torch.cuda.is_available(), 'M1 full CNN training requires CUDA')
    set_seed(int(config['seed']))
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    frames = load_development(root)
    train_count = len(frames['train'])
    frame = __import__('pandas').concat([frames['train'], frames['validation']], ignore_index=True)
    require(train_count == 12152 and len(frame) == 13652, 'certified partition row count differs')
    length = frame['length'].to_numpy(dtype=np.int64)
    labels = frame['label'].to_numpy(dtype=np.int64)
    meta_path = home / 'embeddings/residue_fp16_train_validation.json'
    source = home / 'embeddings/residue_fp16_train_validation.npy'
    offset_file = home / 'embeddings/residue_offsets.npy'
    require(all(p.is_file() for p in [meta_path, source, offset_file]), 'M1 residue embeddings not complete')
    meta = json.loads(meta_path.read_text())
    require(meta.get('schema') == 'amr_v2_m1_residue_embeddings_v1' and
            meta.get('status') == 'complete', 'M1 embedding cache not complete')
    require(meta.get('config_sha256') == sha_bytes(canonical_json(config)), 'M1 embedding config changed')
    require(meta.get('frozen_train_validation_sha256') == SPLIT_HASHES, 'wrong frozen dataset')
    require(meta.get('backbone_commit') == config['backbone_revision'], 'wrong checkpoint')
    require(meta.get('partition_rows') == {'train': train_count, 'validation': 1500}, 'wrong data partitions')
    require(meta.get('partition_ids_sha256') == {k: ids_digest(frames[k]) for k in ('train','validation')},
            'cache ID order differs')
    require(meta.get('all_ids_sha256') == ids_digest(frame), 'overall row order mismatch')
    require(meta.get('file_sha256') == sha256(source), 'M1 cached embedding SHA changed')
    require(meta.get('offsets_file_sha256') == sha256(offset_file), 'offset file SHA changed')
    offsets = np.load(offset_file, allow_pickle=False)
    require(offsets.shape == (len(frame) + 1,) and offsets[0] == 0 and
            np.array_equal(np.diff(offsets), length), 'bad ragged offsets')
    require(meta['shape'] == [int(offsets[-1]), 1280], 'M1 dimensions mismatch')
    require(meta['offsets_values_sha256'] == __import__('hashlib').sha256(offsets.tobytes()).hexdigest(),
            'offsets value hash mismatch')
    local = stage_cache(source, meta['file_sha256'], Path('.cache/m1_cache'))
    cached = np.load(local, mmap_mode='r', allow_pickle=False)
    require(cached.shape == tuple(meta['shape']) and cached.dtype == np.float16, 'invalid local cache')
    print('M1 verified partition data: train', train_count, '| validation', len(frame) - train_count, flush=True)
    print('Checkpoint criterion: validation loss; test/OOD not loaded', flush=True)

    seed = int(config['seed'])
    out = home / 'runs' / f'seed{seed}'
    require_no_partial(out)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out / 'best_model.pt', out / 'last_checkpoint.pt'
    history_path, summary_path = out / 'history.csv', out / 'training_summary.json'
    cfg_copy = out / 'config.json'
    require(not summary_path.exists(), 'M1 already completed; refusing overwrite')
    if cfg_copy.exists():
        require(json.loads(cfg_copy.read_text()) == config, 'M1 run config changed')
    else:
        require(not any(p.exists() for p in [best_path, last_path, history_path]),
                'orphaned M1 training outputs; stop')
        atomic_json(cfg_copy, config)

    network = create_model(config).to(device)
    optimizer = torch.optim.AdamW(network.parameters(),
                                  lr=float(config['learning_rate']),
                                  weight_decay=float(config['weight_decay']))
    criterion = nn.BCEWithLogitsLoss()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    max_batch = int(config['batch_size'])
    tokens = int(config['max_padded_residues_per_batch'])
    val_batches = make_batches(np.arange(train_count, len(frame)), length, max_batch, tokens,
                               shuffle=False, rng=rng)

    epoch_start, best_epoch, no_improvement, best_loss = 1, 0, 0, float('inf')
    history = []
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=True)
        require(state.get('embedding_sha256') == meta['file_sha256'] and
                state.get('config_sha256') == sha_bytes(canonical_json(config)),
                'resume checkpoint references different data/config')
        require(best_path.is_file(), 'cannot resume without best checkpoint')
        network.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        # Ensure optimizer momentum matches GPU device after resume.
        for state_dict in optimizer.state.values():
            for k,v in state_dict.items():
                if torch.is_tensor(v):
                    state_dict[k] = v.to(device)
        epoch_start = int(state['epoch']) + 1
        best_loss = float(state['best_loss'])
        best_epoch = int(state['best_epoch'])
        no_improvement = int(state['patience_count'])
        history = list(state['history'])
        require(len(history) == epoch_start-1, 'checkpoint/history mismatch')
        rng.set_state(state['sampler_rng'])
        torch.set_rng_state(state['cpu_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print('Resuming M1 at epoch', epoch_start, flush=True)
    else:
        require(not best_path.exists() and not history_path.exists(),
                'orphaned training outputs')

    reason = 'max_epochs'
    start_time = time.monotonic()
    for epoch in range(epoch_start, int(config['max_epochs'])+1):
        if no_improvement >= int(config['early_stopping_patience']):
            reason = 'early_stopping'
            break
        epoch_tic = time.monotonic()
        network.train()
        batches = make_batches(np.arange(train_count), length, max_batch, tokens,
                               shuffle=True, rng=rng)
        total_train_loss = 0.0
        for indices in batches:
            x, sizes, y = collate(cached, offsets, length, labels, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(x, sizes)
            loss = criterion(logits, y)
            require(bool(torch.isfinite(loss)), 'nonfinite training loss')
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach().item()) * len(indices)
        val = evaluate(network, cached, offsets, length, labels, val_batches, device)
        require(np.isfinite(val['loss']), 'nonfinite validation loss')
        if val['loss'] < best_loss - float(config['early_stopping_min_delta']):
            best_loss = float(val['loss'])
            best_epoch = epoch
            no_improvement = 0
            atomic_checkpoint(best_path, {
                'epoch': epoch, 'model': network.state_dict(),
                'config_sha256': sha_bytes(canonical_json(config)),
                'embedding_sha256': meta['file_sha256'],
                'backbone_commit': config['backbone_revision'],
                'selection_criterion': 'validation_loss',
            })
        else:
            no_improvement += 1
        history.append({
            'epoch': epoch, 'train_loss': total_train_loss/train_count,
            'validation_loss': val['loss'], 'validation_mcc_0_5': val['mcc'],
            'validation_f1_0_5': val['f1'],
            'validation_average_precision': val['average_precision'],
            'validation_roc_auc': val['roc_auc'],
            'validation_balanced_accuracy_0_5': val['balanced_accuracy'],
            'epoch_seconds': time.monotonic() - epoch_tic,
            'best_epoch_so_far': best_epoch,
            'epochs_without_improvement': no_improvement,
        })
        atomic_checkpoint(last_path, {
            'epoch': epoch, 'model': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config_sha256': sha_bytes(canonical_json(config)),
            'embedding_sha256': meta['file_sha256'],
            'backbone_commit': config['backbone_revision'],
            'best_loss': best_loss, 'best_epoch': best_epoch,
            'patience_count': no_improvement, 'history': history,
            'sampler_rng': rng.get_state(), 'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
        })
        write_history(history_path, history)
        print(f'Epoch {epoch:02d}/{int(config["max_epochs"])}: '
              f'train_loss={total_train_loss/train_count:.5f} val_loss={val["loss"]:.5f} '
              f'val_MCC={val["mcc"]:.4f} best={best_epoch} unimproved={no_improvement}', flush=True)
    if no_improvement >= int(config['early_stopping_patience']):
        reason = 'early_stopping'
    require(best_path.is_file() and last_path.is_file() and history_path.is_file(),
            'missing model checkpoint/history')
    require(best_epoch >= 1, 'best epoch not recorded')
    selected = next(r for r in history if r['epoch'] == best_epoch)
    require(abs(float(selected['validation_loss']) - best_loss) < 1e-10, 'selected checkpoint mismatch')
    atomic_json(summary_path, {
        'status': 'complete', 'dataset_version': 'V2', 'model_id': 'M1',
        'seed': seed, 'train_rows': train_count,
        'validation_rows': len(frame)-train_count,
        'checkpoint_metric': 'validation_loss',
        'selection_uses_test_or_ood': False,
        'best_epoch': best_epoch, 'epochs_executed': len(history),
        'best_validation_loss': best_loss,
        'best_validation_metrics_at_0_5': {
            'mcc': selected['validation_mcc_0_5'], 'f1': selected['validation_f1_0_5'],
            'roc_auc': selected['validation_roc_auc'],
            'average_precision': selected['validation_average_precision'],
            'balanced_accuracy': selected['validation_balanced_accuracy_0_5'],
        },
        'stop_reason': reason, 'elapsed_seconds': time.monotonic()-start_time,
        'config_sha256': sha_bytes(canonical_json(config)),
        'embedding_sha256': meta['file_sha256'],
        'backbone_commit': config['backbone_revision'],
        'audit_report_sha256': meta['audit_report_sha256'],
        'best_model_sha256': sha256(best_path),
        'last_checkpoint_sha256': sha256(last_path),
        'history_sha256': sha256(history_path),
    })
    print('M1 TRAINING COMPLETE; best epoch:', best_epoch, flush=True)
    print('Summary:', summary_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    home = Path(__file__).resolve().parent
    config = json.loads((home/'config.json').read_text())
    require(config['model_id'] == 'M1' and config['checkpoint_metric'] == 'validation_loss',
            'M1 configuration invalid')
    if args.smoke:
        smoke(config)
    else:
        train(args.root, config, home)

if __name__ == '__main__':
    main()
