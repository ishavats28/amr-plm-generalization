"""Frozen M5 residue-cache verification and local staging for M6.

No training/evaluation data outside train+validation is ever opened.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from common import (SPLIT_HASHES, canonical_json, ids_digest, load_development,
                    require, sha256, sha_bytes)

M5_CACHE_SHA = 'e2ced9942c9dc76058eda8db22f8ea860494cb0743b021a7c1713b22c7a99d8b'
M5_PATH = Path('training_v2/M5_protbert_multiscale_cnn')


def inspect_cache(root: Path, cfg: dict, *, hash_full: bool = False):
    root = Path(root)
    data = load_development(root)
    frame = pd.concat([data['train'], data['validation']], ignore_index=True)
    ntrain = len(data['train'])
    require(ntrain == 12152 and len(frame) == 13652, 'unexpected certified training and validation counts')
    m5home = root / M5_PATH
    cfg_m5 = json.loads((m5home / 'config.json').read_text(encoding='utf-8'))
    meta_path = m5home / 'embeddings/residue_fp16_train_validation.json'
    source = m5home / 'embeddings/residue_fp16_train_validation.npy'
    offs_file = m5home / 'embeddings/residue_offsets.npy'
    for p in [meta_path, source, offs_file]:
        require(p.is_file(), f'M5 cache component missing: {p}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    require(meta.get('schema') == 'amr_v2_m5_protbert_residue_embeddings_v1' and meta.get('status') == 'complete',
            'M5 source cache not complete')
    require(meta.get('partition_order') == ['train', 'validation'], 'cache partition order mismatch')
    require(meta.get('partition_rows') == {'train':12152, 'validation':1500}, 'cache row counts mismatch')
    require(meta.get('partition_ids_sha256') == {k: ids_digest(data[k]) for k in ('train','validation')},
            'M5 residue cache partition ID order mismatch')
    require(meta.get('all_ids_sha256') == ids_digest(frame), 'M5 cache overall ID order mismatch')
    require(meta.get('frozen_train_validation_sha256') == SPLIT_HASHES, 'M5 cache sources differ')
    require(meta.get('backbone_commit') == cfg['backbone_revision'] == cfg_m5['backbone_revision'],
            'ProtBERT revision mismatch')
    require(meta.get('config_sha256') == sha_bytes(canonical_json(cfg_m5)), 'M5 extraction config was modified')
    require(meta.get('file_sha256') == M5_CACHE_SHA, 'M5 residue cache expected SHA mismatch')
    require(meta.get('backbone') == cfg['backbone'] == 'Rostlab/prot_bert',
            'ProtBERT source/cache backbone mismatch')
    require(meta.get('audit_report_sha256') == sha256(root/'metadata/v2_split_audit.json'),
            'cache audit provenance mismatch')
    require(meta.get('shape') == [4229962, 1024] and meta.get('dtype') == 'float16', 'M5 cache shape/type metadata unexpected')
    require(cfg['residue_embedding_source'] == str(M5_PATH/'embeddings/residue_fp16_train_validation.npy'),
            'M6 configuration points to unexpected embedding source')
    require(not (m5home/'embeddings/extract_progress.json').exists(), 'M5 extraction progress file remains')
    lengths = frame['length'].to_numpy(dtype=np.int64)
    offsets = np.load(offs_file, allow_pickle=False)
    require(offsets.shape == (len(frame)+1,) and int(offsets[0]) == 0 and
            np.array_equal(np.diff(offsets), lengths) and int(offsets[-1]) == 4229962,
            'ragged offsets differ from certified proteins')
    require(sha256(offs_file) == meta.get('offsets_file_sha256'), 'M5 offsets checksum mismatch')
    require(hashlib.sha256(offsets.tobytes()).hexdigest() == meta.get('offsets_values_sha256'),
            'M5 offset value checksum mismatch')
    require(source.stat().st_size >= 8*2**30, 'unexpectedly small M5 residue cache')
    if hash_full:
        print('Rechecking M5 residue cache SHA256 on Drive...', flush=True)
        require(sha256(source) == M5_CACHE_SHA, 'M5 cache content hash differs')
    return frame, ntrain, lengths, offsets, meta, source


def stage_cache(source: Path, expected: str, *, stage: Path = Path('.cache/m5_cache/m5_residue_fp16_train_validation.npy')):
    """Reuse verified local M5 cache where present; else copy to a local cache.

    This deliberately requires the file hash, even on a reused local copy.
    """
    stage.parent.mkdir(parents=True, exist_ok=True)
    if stage.exists():
        require(stage.is_file() and sha256(stage) == expected, 'previous local M5 stage hash mismatch')
        print('Reusing verified M5 cache staged locally', flush=True)
        return stage
    size = source.stat().st_size
    require(shutil.disk_usage(stage.parent).free > int(size*1.15),
            f'Insufficient local free space to stage {size / 2**30:.2f} GiB cache')
    temp = stage.with_name(stage.name + '.partial')
    require(not temp.exists(), 'incomplete stage file exists; inspect manually')
    print(f'Staging {size / 2**30:.2f} GiB cache from source to local cache...', flush=True)
    with source.open('rb') as inp, temp.open('xb') as out:
        while chunk := inp.read(16 * 1024 * 1024):
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    require(sha256(temp) == expected, 'local staged cache checksum differs')
    os.replace(temp, stage)
    return stage
