#!/usr/bin/env python3
"""One-time immutable ProtBERT revision pinning; does not load any dataset."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import atomic_json, require

def is_sha(x):
    return isinstance(x, str) and len(x) == 40 and all(c in '0123456789abcdef' for c in x.lower())

def resolve_repo_sha(name):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id=name, revision='main')
    return getattr(info, 'sha', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    require(home == args.root.resolve()/'training_v2/M4_protbert_mlp', 'wrong M4 project root')
    path = home/'config.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    require(cfg['model_id'] == 'M4' and cfg['backbone'] == 'Rostlab/prot_bert', 'unexpected model config')
    for subfolder in ('embeddings', 'runs', 'logs'):
        dest = home/subfolder
        require(not dest.exists() or not any(dest.iterdir()),
                f'M4 {subfolder} has artifacts; refusing to change pinned backbone')
    revision = cfg['backbone_revision']
    if is_sha(revision):
        print('M4 ProtBERT already pinned:', revision, flush=True)
        return
    require(revision == 'main', 'unexpected unpinned revision')
    commit = resolve_repo_sha(cfg['backbone'])
    require(is_sha(commit), 'Hugging Face did not provide an immutable 40-character commit')
    cfg['backbone_revision'] = commit.lower()
    atomic_json(path, cfg)
    print('M4 ProtBERT immutable revision pinned:', commit.lower(), flush=True)
    print('Do not modify M4 config after extraction begins.', flush=True)

if __name__ == '__main__':
    main()
