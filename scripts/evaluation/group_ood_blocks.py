#!/usr/bin/env python3
"""Original 24-block Group-OOD analysis from the study notebook.

Requires the locked split audit, held-out prediction records, and source manifests.
Writes to a new output directory outside the source study root.
"""
from pathlib import Path
import argparse
import hashlib, json, os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True, help="Original study root")
parser.add_argument("--output-dir", type=Path, required=True, help="New result directory outside the source root")
args = parser.parse_args()
R = args.root.resolve()
O = args.output_dir.resolve()
T = O.with_name(O.name + ".partial")
if O == R or R in O.parents:
    parser.error("--output-dir must be outside the source study root")

def verify(ok, why):
    if not bool(ok):
        raise RuntimeError('STOP: ' + why)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

print('HYPOSEQ | ORIGINAL 24-BLOCK GROUP-OOD ANALYSIS')

verify(
    R.is_dir() and not O.exists() and not T.exists(),
    'Drive missing or an earlier result/partial folder exists'
)

A = R/'metadata/v2_split_audit.json'
M = R/'metadata/v2_split_manifest.json'

verify(
    sha(A) ==
    'f587a237fdf4a48b1beaa559cdd56e2461cf46b600b1038e3c5969eb29a6e8ac',
    'Split audit hash mismatch'
)

audit = read(A)
man = read(M)

verify(
    sha(M) == audit['manifest_sha256_before_audit'],
    'Split manifest hash mismatch'
)

for rel in (
    'data/processed/ood.tsv.gz',
    'metadata/v2_biological_groups.tsv.gz'
):
    verify(
        sha(R/rel) ==
        man['output_hashes'][rel] ==
        audit['verified_output_hashes'][rel],
        'Certified data hash mismatch: ' + rel
    )

df = pd.read_csv(
    R/'data/processed/ood.tsv.gz',
    sep='\t',
    dtype={'protein_id':str, 'homology_component':str}
)

g = pd.read_csv(
    R/'metadata/v2_biological_groups.tsv.gz',
    sep='\t',
    dtype={'homology_component':str, 'ood_block_id':str}
)

verify(
    len(df) == 642 and df.protein_id.is_unique
    and df.label.isin([0,1]).all()
    and df.label.sum() == 321
    and df.split.eq('ood').all(),
    'OOD IDs/labels invalid'
)

verify(
    g.homology_component.is_unique
    and df.homology_component.notna().all(),
    'Homology-component IDs missing or duplicated'
)

verify(
    df.groupby('homology_component').label.nunique().eq(1).all(),
    'Mixed-label strict homology component'
)

B = man['ood_blocks']

verify(
    len(B) == 24
    and len(audit['ood_reconstruction']['blocks']) == 24,
    'Expected 24 source-backed blocks'
)

block_map = {
    b['block_id']:set(b['components']) for b in B
}

verify(
    len(block_map) == 24
    and sum(len(x) for x in block_map.values()) == 29
    and len(set().union(*block_map.values())) == 29
    and {frozenset(x) for x in block_map.values()} ==
        {frozenset(x) for x in audit['ood_reconstruction']['blocks']},
    'Block definitions disagree with original audit'
)

verify(
    sum(int(b['positive_proteins']) for b in B) == 321
    and max(int(b['positive_proteins']) for b in B) == 170,
    'Unexpected original block sizes'
)

g = g.set_index('homology_component')

verify(
    set(df.homology_component).issubset(set(g.index)),
    'Missing grouping metadata for OOD component'
)

pos = df.loc[
    df.label.eq(1),
    ['protein_id','homology_component']
].copy()

verify(
    set(pos.homology_component) ==
    set().union(*block_map.values()),
    'Original positive component membership mismatch'
)

verify(
    g.loc[pos.homology_component.unique(),'partition'].eq('ood').all(),
    'Positive grouping partition differs'
)

verify(
    g.loc[
        df.loc[df.label.eq(0),'homology_component'].unique(),
        'ood_block_id'
    ].isna().all(),
    'Negative components have positive-block assignments'
)

pos['block_id'] = pos.homology_component.map(g['ood_block_id'])

verify(
    pos.block_id.notna().all()
    and set(pos.block_id) == set(block_map),
    'Missing or unexpected positive block assignments'
)

blocks = []

for b in B:
    bid = b['block_id']
    p = pos.loc[pos.block_id.eq(bid)]

    verify(
        len(p) == int(b['positive_proteins'])
        and set(p.homology_component) == block_map[bid]
        and pd.to_numeric(
            g.loc[sorted(block_map[bid]),'positives']
        ).sum() == len(p),
        'Block membership/size mismatch: ' + bid
    )

    blocks.append({
        'block_id':bid,
        'positive_proteins':len(p),
        'strict_components':len(block_map[bid]),
        'component_ids':';'.join(sorted(block_map[bid])),
        'share_of_all_ood_positives':len(p)/321
    })

print(
    'ORIGINAL BLOCK MEMBERSHIP: PASS '
    '(24 blocks, 29 components, 321 positives)'
)

P = R/'evaluation_v2/phase2_holdout_predictions_v1'
S = P/'FINAL_INFERENCE_MANIFEST.json'

verify(
    sha(S) ==
    '226252c29f1d86b1cba6c6a42d1ae87bdf41c68098d52b475a8b0e2f95ffbb47',
    'Locked held-out prediction summary hash mismatch'
)

s = read(S)

verify(
    len(s['results']) == 12
    and sha(
        R/'evaluation_v2/inference_lock_v1/INFERENCE_LOCK_MANIFEST.json'
    ) == s['inference_lock_manifest_sha256']
    and sha(
        R/'evaluation_v2/phase2_protocol_lock_v1/PROTOCOL.md'
    ) == 'edf7610cd6b2548c8365bee5d24b8c930f5b2eac870f73b0df4433a00a830421',
    'Inference/protocol lock differs'
)

entries = {
    (v['partition'],v['model'],v['regime']):v
    for v in s['results']
}

verify(len(entries) == 12, 'Duplicate inference records')

per_block = []
summaries = []
components = []

for model in ('M1','M3','M0'):
    for regime in ('frozen','LoRA'):

        name = f'ood_{model}_{regime}'
        folder = P/name

        rec = read(folder/'RESULT.json')
        entry = entries[('ood',model,regime)]

        verify(
            all(
                rec[k] == entry[k]
                for k in (
                    'predictions_csv_sha256',
                    'checkpoint_sha256',
                    'input_sha256',
                    'metrics'
                )
            )
            and rec['completed'] is True
            and rec['threshold'] == 0.5
            and rec['n'] == 642
            and rec['input_sha256'] ==
                man['output_hashes']['data/processed/ood.tsv.gz'],
            'Locked result record mismatch: ' + name
        )

        path = folder/'predictions.csv'

        verify(
            sha(path) == rec['predictions_csv_sha256'],
            'Prediction SHA mismatch: ' + name
        )

        pr = pd.read_csv(
            path,
            dtype={'protein_id':str,'homology_component':str}
        )

        verify(
            len(pr) == 642
            and pr.protein_id.is_unique
            and set(pr.columns) == {
                'protein_id','true_label','homology_component',
                'probability_amr','prediction_at_0_5'
            },
            'Prediction file columns/IDs invalid: ' + name
        )

        v = df[
            ['protein_id','homology_component','label']
        ].merge(
            pr,
            on='protein_id',
            validate='one_to_one',
            suffixes=('_src','_pred')
        )

        prob = v.probability_amr.to_numpy(dtype=float)

        verify(
            len(v) == 642
            and v.homology_component_src.eq(
                v.homology_component_pred
            ).all()
            and v.label.eq(v.true_label).all()
            and np.isfinite(prob).all()
            and ((prob >= 0) & (prob <= 1)).all()
            and np.array_equal(
                v.prediction_at_0_5.to_numpy(dtype=int),
                (prob >= 0.5).astype(int)
            ),
            'Prediction/label alignment or threshold mismatch: ' + name
        )

        yes = v.loc[
            v.label.eq(1),
            ['protein_id','homology_component_src','prediction_at_0_5']
        ]

        joined = pos.merge(
            yes[['protein_id','prediction_at_0_5']],
            on='protein_id',
            validate='one_to_one'
        )

        TP = int(joined.prediction_at_0_5.sum())
        FN = 321 - TP

        verify(
            TP == int(rec['metrics']['tp'])
            and FN == int(rec['metrics']['fn']),
            'TP/FN differ from locked results: ' + name
        )

        start = len(per_block)

        for b in B:
            bid = b['block_id']
            sub = joined.loc[joined.block_id.eq(bid)]

            tp = int(sub.prediction_at_0_5.sum())
            fn = len(sub) - tp

            per_block.append({
                'model':model,
                'regime':regime,
                'block_id':bid,
                'positive_proteins':len(sub),
                'strict_components':len(block_map[bid]),
                'tp':tp,
                'fn':fn,
                'positive_recall':tp/len(sub),
                'share_of_all_false_negatives':fn/FN if FN else None
            })

        bsub = per_block[start:]
        recalls = np.array([r['positive_recall'] for r in bsub])

        big = next(
            r for r in bsub
            if r['block_id'] == 'OOD_BLOCK_001'
        )

        verify(
            big['positive_proteins'] == 170
            and sum(r['fn'] for r in bsub) == FN,
            'Block false-negative totals differ: ' + name
        )

        summaries.append({
            'model':model,
            'regime':regime,
            'overall_positive_recall':TP/321,
            'macro_block_recall':float(recalls.mean()),
            'median_block_recall':float(np.median(recalls)),
            'blocks_with_zero_recall':int((recalls==0).sum()),
            'blocks_with_full_recall':int((recalls==1).sum()),
            'tp':TP,
            'fn':FN,
            'largest_block_fn':big['fn'],
            'largest_block_share_of_fn':big['fn']/FN if FN else None
        })

        # Component summaries are separate from positive biological blocks.
        # Negative components receive specificity, not an invented block ID.
        for cid, comp in v.groupby('homology_component_src'):
            label = int(comp.label.iloc[0])
            pred = comp.prediction_at_0_5.to_numpy(dtype=int)

            components.append({
                'model':model,
                'regime':regime,
                'homology_component':cid,
                'label':label,
                'n_proteins':len(comp),
                'positive_block_id':(
                    g.loc[cid,'ood_block_id'] if label else None
                ),
                'positive_recall':(
                    float(pred.mean()) if label else None
                ),
                'negative_specificity':(
                    float((1-pred).mean()) if not label else None
                )
            })

        print(
            f'{name}: recall={TP/321:.4f}, '
            f'block_macro={recalls.mean():.4f}, '
            f'largest-block FN={big["fn"]}/{FN}',
            flush=True
        )

verify(
    len(per_block) == 144
    and len(summaries) == 6
    and len(components) == 6*df.homology_component.nunique(),
    'Incomplete outputs'
)

# Write only after all source and prediction checks pass.
T.mkdir(parents=True,exist_ok=False)

for name, rows in {
    'OOD_VERIFIED_BLOCK_DEFINITIONS.csv':blocks,
    'OOD_PER_BLOCK_MODEL_RECALL.csv':per_block,
    'OOD_MODEL_BLOCK_SUMMARY.csv':summaries,
    'OOD_STRICT_COMPONENT_SUMMARY.csv':components,
}.items():
    pd.DataFrame(rows).to_csv(
        T/name,
        index=False,
        float_format='%.12g'
    )

record = {
    'status':'complete',
    'threshold':0.5,
    'analysis':'source-backed positive block recall',
    'blocks':24,
    'positive_components':29,
    'positive_proteins':321,
    'largest_block':'OOD_BLOCK_001',
    'largest_block_n':170,
    'source_sha256':{
        'audit':sha(A),
        'manifest':sha(M),
        'ood':sha(R/'data/processed/ood.tsv.gz'),
        'groups':sha(R/'metadata/v2_biological_groups.tsv.gz'),
        'inference_summary':sha(S)
    },
    'constraint':(
        'No negative proteins assigned positive blocks; '
        'no block MCC/AUROC/specificity'
    )
}

(T/'ANALYSIS_RECORD.json').write_text(
    json.dumps(record,indent=2,sort_keys=True)+'\n'
)

hashes = {
    p.name:sha(p)
    for p in T.iterdir() if p.is_file()
}

(T/'OUTPUT_HASHES.json').write_text(
    json.dumps(hashes,indent=2,sort_keys=True)+'\n'
)

verify(
    all(sha(T/k) == v for k,v in hashes.items()),
    'Output integrity failed'
)

os.replace(T,O)

print('\nGROUP-OOD BIOLOGICAL-BLOCK ANALYSIS: COMPLETE')
print('Saved:',O)
print('OUTPUT_HASHES SHA256:',sha(O/'OUTPUT_HASHES.json'))
print('No model reruns; original data/predictions unmodified.')
print(pd.DataFrame(summaries).to_string(index=False))
