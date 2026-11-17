#!/usr/bin/env python3
"""Validation-only inference-policy checks before held-out evaluation.

--self-test:  CPU-only deterministic schedule, pooled-feature and alignment tests.
--validate:   Re-encode the certified validation partition only with a frozen, partition-local batch policy,
              then score the three frozen heads in FP32 and compare to archived
              validation results. Independently verify the existing six replay
              evidence archives. Not a held-out evaluator, not an inference lock.

The 3 selected LoRA models already passed full live validation replay. This script
verifies their immutable evidence archives rather than needlessly rerunning them.
This script contains NO option to read test, OOD or ESKAPE records.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil

import numpy as np

LOCK_CANDIDATES_SHA = '24deeeaa46ba20ad5693b4d874e4f9a29424034fd70d75b5fcbf82cdf34b842c'
M1_EXTRACT_SHA = '05edcd717f654f35ec91a12ebd669e95ab2d3381a078b0804cdc386874e54415'
EVIDENCE = {
    'HypoSeq_M1_Validation_Evidence_v1.zip': '5ed58214b0b2f9b0b43a8de8ab24ea946ad6403b016227c1230cafcc521bc2d5',
    'HypoSeq_M3_LoRA_Validation_Evidence_v1.zip': '3718bc4b006f38c1c630e62afb688953be3bb8a1bf7681719e70f238c5d911c0',
    'HypoSeq_M0_LoRA_Validation_Evidence_v1.zip': '73bb490721f2178710903a3d03b0ac923322c4ebf9bdd950ccb190771e693dc7',
    'HypoSeq_M3_Frozen_Validation_Evidence_v1.zip': '1a13cb8f2ef1cac2574d7b023c8a1c7c657c1a878856eac3956e9cc10a05a43f',
    'HypoSeq_M0_Frozen_Validation_Evidence_v1.zip': 'c5f2e2e9341afddd7d5ee142a8b10e94ba1b46ddc2fc88090d739cdae7759500',
}
# This is a development-stage *proposal* for a deterministic prospective inference policy.
# Held-out access remains forbidden. When deployed, each unseen split is sorted
# only within itself, NEVER using train/validation sequence context or labels.
POLICY = {
    'schema': 'hyposeq_v2_unseen_sequence_policy_candidate_v1',
    'status': 'unlocked_validation_candidate_only',
    'source': 'original_M1_extractor_and_original_frozen_heads',
    'input_partition_order': 'stable_ascending_length_within_partition',
    'encoder_max_sequences_per_batch': 4,
    'encoder_max_padded_tokens_per_batch': 2048,
    'encoder_dtype': 'bfloat16',
    'encoder_autocast_dtype': 'bfloat16',
    'residue_dtype_after_encoder': 'float16',
    'exclude_special_and_padding_tokens': True,
    'truncation': False,
    'frozen_M1_M3_head_dtype': 'float32',
    'frozen_M0_pooling': 'mean_FP32_of_unrounded_BF16_residues_then_FP16_storage_FP32_head',
    'frozen_M0_head_dtype': 'float32',
    'lora_validation_evidence': 'previous_exact_live_replays_archive_hashes',
    'fixed_threshold': 0.5,
    'batch_policy_separate_for_each_new_partition': True,
    'test_or_ood_inference_authorized': False,
}
OUT = Path('validation_policy_output')
STAGE = Path('.validation_policy_inprogress')

def require(test, reason):
    if not bool(test):
        raise RuntimeError('STOP: ' + reason)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def make_length_batches(lengths, max_items=4, token_budget=2048):
    lengths = np.asarray(lengths, dtype=np.int64)
    require(len(lengths)>0 and bool(np.all((lengths>=1)&(lengths<=1014))), 'Invalid length vector')
    order = np.argsort(lengths,kind='stable')
    result=[]; cursor=0
    while cursor < len(order):
        seqs=[]; longest=0
        while cursor+len(seqs)<len(order) and len(seqs)<max_items:
            idx=int(order[cursor+len(seqs)])
            proposed=max(longest,int(lengths[idx])+2)
            if seqs and proposed*(len(seqs)+1)>token_budget: break
            seqs.append(idx); longest=proposed
        require(bool(seqs) and longest*len(seqs)<=token_budget,'Batch planner overflow')
        result.append(np.asarray(seqs,dtype=np.int64));cursor+=len(seqs)
    flat=np.concatenate(result)
    require(np.array_equal(flat,order), 'Unstable or incomplete batch indices')
    return result

def self_test():
    lens=np.asarray([161,19,41,250,41,2,1014,100,19,900],dtype=np.int64)
    a=make_length_batches(lens)
    b=make_length_batches(lens.copy())
    require(len(a)==len(b) and all(np.array_equal(x,y) for x,y in zip(a,b)),
            'Batch schedule is nondeterministic')
    require(np.array_equal(np.concatenate(a),np.argsort(lens,kind='stable')),
            'Stable order test failed')
    for batch in a:
        require(len(batch)<=4 and (max(lens[batch])+2)*len(batch)<=2048,
                'Token budget test failed')
    residue=np.array([[1,2],[3,5],[5,10]],dtype=np.float16)
    pooled=np.asarray(residue,dtype=np.float32).mean(0).astype(np.float16)
    require(np.allclose(pooled,np.array([3.,17./3.],dtype=np.float16)),
            'Frozen pooled representation test failed')
    require(POLICY['test_or_ood_inference_authorized'] is False, 'Policy boundary absent')
    print('SYNTHETIC STABLE BATCH / FP16 STORAGE / FP32 POOL: PASS')
    print('No project files opened. No research or holdout files accessed.')

def verified_evidence(root):
    import zipfile
    where=root/'evaluation_v2'
    for name,expected in EVIDENCE.items():
        path=where/name
        require(path.is_file() and digest(path)==expected,f'Archived replay evidence changed/missing: {name}')
        with zipfile.ZipFile(path) as z:
            require(z.testzip() is None,f'Invalid evidence archive: {name}')
            internal=read_json_from_zip(z,'EVIDENCE_MANIFEST.json')
            for rel,sha in internal.items():
                require(hashlib.sha256(z.read(rel)).hexdigest()==sha,
                        f'Evidence entry hash differs: {name}: {rel}')
        print('ARCHIVED EVIDENCE PASS:',name,flush=True)

def read_json_from_zip(z,relative):
    return json.loads(z.read(relative).decode('utf-8'))

def encode_original_precision_with_pooled(sequences,tokenizer,encoder):
    """M1's original tokenization/residue FP16 path plus M0's FP32 pre-storage pooling."""
    import torch
    payload=tokenizer(sequences,padding=True,truncation=False,
                      return_special_tokens_mask=True,return_tensors='pt')
    input_ids=payload['input_ids'].to('cuda')
    attention=payload['attention_mask'].to('cuda')
    special=payload['special_tokens_mask'].to('cuda')
    require(input_ids.shape[1]<=1024,'Tokenized length exceeds ESM-2 limit')
    mask=attention.bool() & ~special.bool()
    require(mask.sum(1).cpu().tolist()==[len(s) for s in sequences],
            'Special-token or residue masking differs from original')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=encoder(input_ids=input_ids,attention_mask=attention).last_hidden_state
        # Store residue embeddings exactly as certified M1 extractor does.
        residue=[hidden[i,mask[i]].to(torch.float32).cpu().numpy().astype(np.float16)
                 for i in range(len(sequences))]
        # For M0, mean in FP32 BEFORE the final FP16 storage conversion.
        pooled=[hidden[i,mask[i]].to(torch.float32).mean(dim=0).cpu().numpy().astype(np.float16)
                for i in range(len(sequences))]
    require(all(v.shape==(len(s),1280) and np.isfinite(v).all()
                for v,s in zip(residue,sequences)),'Invalid original residue encoding')
    require(all(v.shape==(1280,) and np.isfinite(v).all() for v in pooled),
            'Invalid FP32 pooled embedding')
    return residue,pooled

def original_predictions(root,model,validation_ids,y):
    import pandas as pd
    if model in ('M1','M3'):
        folder=root/'benchmark_v2_validation_addendum'/model
        manifest=read_json(folder/'MANIFEST.json')
        summary=read_json(root/'benchmark_v2_validation_addendum/summary/FINAL_MANIFEST.json')
        require(digest(folder/'MANIFEST.json')==summary['model_record_sha256'][model],
                'Archived frozen addendum differs')
        rel='validation_predictions.csv.gz'
        require(digest(folder/rel)==manifest['output_hashes'][rel],
                f'Archived {model} predictions changed')
        p=pd.read_csv(folder/rel,dtype={'protein_id':str})
    else:
        reference=read_json(root/'benchmark_v2_freeze/existing_m0_validation_analysis.json')
        manifest_path=root/reference['original_relative_path']
        require(digest(manifest_path)==reference['sha256'],'Frozen M0 original analysis manifest differs')
        manifest=read_json(manifest_path)
        rel='validation_probabilities.csv.gz'
        require(digest(manifest_path.parent/rel)==manifest['output_hashes'][rel],
                'Frozen M0 archived predictions changed')
        p=pd.read_csv(manifest_path.parent/rel,dtype={'protein_id':str})
    require(len(p)==len(validation_ids) and p['protein_id'].tolist()==validation_ids,
            f'{model}: original validation ID order mismatch')
    require(p['true_label'].to_numpy(dtype=np.int64).tolist()==y.tolist(),
            f'{model}: original validation labels differ')
    return p['probability_amr'].to_numpy(dtype=np.float64)

def print_diffs(model,y,old,new):
    from sklearn.metrics import matthews_corrcoef,f1_score,roc_auc_score,average_precision_score
    names=('MCC','F1','AUROC','AP','Brier')
    def scores(p):
        lab=(p>=.5).astype(np.int64)
        return [float(matthews_corrcoef(y,lab)),float(f1_score(y,lab)),
                float(roc_auc_score(y,p)),float(average_precision_score(y,p)),
                float(np.mean((p-y)**2))]
    ref=scores(old);got=scores(new)
    print(f'\n{model} FROZEN | ARCHIVED vs PARTITION-LOCAL POLICY (development ONLY)')
    for k,x,z in zip(names,ref,got):
        print(f'  {k:<7} original={x:.9f} candidate={z:.9f} delta={z-x:+.9f}')
    flips=np.flatnonzero((old>=.5)!=(new>=.5))
    print(f'  changed_predictions={len(flips)}, mean_absolute_probability_delta={float(np.mean(np.abs(old-new))):.9f}')
    return {'original_metrics':dict(zip(names,ref)),
            'candidate_metrics':dict(zip(names,got)),
            'classification_flips':int(len(flips)),
            'mean_absolute_probability_difference':float(np.mean(np.abs(old-new))),
            'maximum_absolute_probability_difference':float(np.max(np.abs(old-new)))}

def run(root):
    import pandas as pd
    import torch
    from sklearn.metrics import confusion_matrix
    import hyposeq_inference_validation_v1 as base

    root=Path(root)
    require(not OUT.exists() and not STAGE.exists(),
            f'Existing diagnostic output/stage found: {OUT} or {STAGE}. Inspect, do not overwrite.')
    require(digest(Path(base.__file__))==
            '71c8f971ba509e4d56ed455d2b187ca5c5197f673d08d291038dc623084e8d0c',
            'Base inference source does not match validated version')
    inv,proto=base.verified_preflight(root)
    base.verify_checkpoint_structures(root,inv)
    base.runtime_guard(proto)
    verified_evidence(root)
    extract_path=root/'training_v2/M1_esm2_multiscale_cnn/extract.py'
    require(digest(extract_path)==M1_EXTRACT_SHA,'Original M1 extractor source changed')
    sys.path.insert(0,str(root/'training_v2'))
    from common import load_development
    frames=load_development(root)  # certified TRAIN and VALIDATION only
    val=frames['validation']; n=len(val)
    require(n==1500 and val['protein_id'].is_unique,'Invalid certified validation rows')
    labels=val['label'].to_numpy(dtype=np.int64)
    require(int(labels.sum())==750,'Validation class count mismatch')
    lengths=val['length'].to_numpy(dtype=np.int64)
    seq=val['sequence'].astype(str).tolist()
    ids=val['protein_id'].astype(str).tolist()
    schedule=make_length_batches(lengths)
    print(f'PROPOSED UNSEEN POLICY: 4 seq / 2048 tokens, partition-local stable length sort, {len(schedule)} batches',flush=True)
    print('No train samples included in new inference batches. No test/OOD reads.',flush=True)
    ext=import_file('hyposeq_verified_original_M1_extractor',extract_path)
    cfg=read_json(root/'training_v2/M1_esm2_multiscale_cnn/config.json')
    require(cfg['embedding_max_sequences_per_batch']==4 and
            cfg['embedding_max_padded_tokens_per_batch']==2048,
            'Original frozen encoder batching settings changed')
    # Compare our schedule with exact original function plan_batch applied to this
    # within-validation order; independently prevents a new batching algorithm.
    order=np.argsort(lengths,kind='stable')
    cursor=0
    for batch in schedule:
        original=ext.plan_batch(order,cursor,lengths,4,2048)
        require(np.array_equal(batch,np.asarray(original,dtype=np.int64)),
                'Proposed batching diverges from original extraction algorithm')
        cursor+=len(batch)
    require(cursor==n,'Original schedule failed to cover partition')
    STAGE.mkdir(parents=True,exist_ok=False)
    scratch=STAGE/'validation_residue_fp16.npy'
    offset=np.zeros(n+1,dtype=np.int64)
    np.cumsum(lengths,out=offset[1:])
    residue=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float16,
                                     shape=(int(offset[-1]),1280))
    pooled=np.zeros((n,1280),dtype=np.float16)
    print(f'LOCAL scratch: {scratch.stat().st_size/2**30:.2f} GiB',flush=True)
    tokenizer,enc=ext.load_encoder(cfg)
    for i,batch in enumerate(schedule,1):
        these_sequences=[seq[j] for j in batch]
        vectors, pooled_vectors=encode_original_precision_with_pooled(these_sequences,tokenizer,enc)
        if i==1:
            # Compare our shared single-forward procedure directly with the verified
            # immutable original extractor, before trusting it for the entire split.
            reference_vectors=ext.encode_residues(these_sequences,tokenizer,enc)
            require(all(np.array_equal(a,b) for a,b in zip(vectors,reference_vectors)),
                    'New combined encoding differs from original extractor')
            print('Original M1 extractor FP16 parity on first batch: PASS',flush=True)
        require(len(vectors)==len(batch),'Original encoder returned incorrect batch count')
        for j,features,pool in zip(batch,vectors,pooled_vectors):
            require(features.shape==(int(lengths[j]),1280) and features.dtype==np.float16 and
                    np.isfinite(features).all(), 'Nonfinite or malformed FP16 protein features')
            residue[offset[j]:offset[j+1]]=features
            pooled[j]=pool
        if i%50==0 or i==len(schedule):
            print(f'  Frozen features: {i}/{len(schedule)} batches',flush=True)
    residue.flush(); del residue,enc,tokenizer
    torch.cuda.empty_cache()
    cache=np.load(scratch,mmap_mode='r',allow_pickle=False)
    print('Frozen validation feature generation: COMPLETE',flush=True)
    inv_by_name={(c['model'],c['regime']):c for c in inv['candidates']}
    report={'policy':POLICY,'original_evidence_archives_sha256':EVIDENCE,
            'base_inference_source_sha256':digest(Path(base.__file__)),
            'original_extractor_sha256':M1_EXTRACT_SHA,
            'candidates_lock_sha256':LOCK_CANDIDATES_SHA,
            'partition':'validation','n':n,'encoder_batches':len(schedule),
            'test_ood_opened':False,'frozen_models':{}}
    for name in ('M1','M3','M0'):
        c=inv_by_name[(name,'frozen')]
        cfg=read_json(root/base.HOME[name]/'config.json')
        head=base.make_head(root,name,cfg).to('cuda:0')
        saved=torch.load(root/c['checkpoint_relative_path'],map_location='cpu',weights_only=True)
        require(saved['epoch']==c['selected_epoch'],'Frozen checkpoint selected epoch mismatched')
        head.load_state_dict(saved['model'],strict=True)
        head.eval();del saved
        predictions=np.full(n,np.nan,dtype=np.float64)
        assigned=np.zeros(n,dtype=np.int8)
        if name=='M0':
            batches=[np.arange(i,min(i+int(cfg['batch_size']),n),dtype=np.int64)
                     for i in range(0,n,int(cfg['batch_size']))]
        else:
            folder=root/base.HOME[name]
            sys.path.insert(0,str(folder))
            for key in ('model','cache'):
                sys.modules.pop(key,None)
            trainer=import_file(f'hyposeq_original_{name}_frozen_trainer_for_batches',folder/'train.py')
            batches=trainer.make_batches(np.arange(n,dtype=np.int64),lengths,
                    int(cfg['batch_size']),int(cfg['max_padded_residues_per_batch']),
                    shuffle=False,rng=torch.Generator(device='cpu').manual_seed(42))
        with torch.inference_mode():
            for batch in batches:
                if name=='M0':
                    x=torch.from_numpy(pooled[batch].astype(np.float32)).to('cuda:0')
                    logits=head(x)
                else:
                    pad=np.zeros((len(batch),int(lengths[batch].max()),1280),dtype=np.float32)
                    for row,idx in enumerate(batch):
                        pad[row,:int(lengths[idx])]=cache[offset[idx]:offset[idx+1]]
                    x=torch.from_numpy(pad).to('cuda:0')
                    sizes=torch.from_numpy(lengths[batch].astype(np.int64)).to('cuda:0')
                    logits=head(x,sizes)
                require(logits.shape==(len(batch),) and bool(torch.isfinite(logits).all()),
                        f'{name}: malformed classifier output')
                p=torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
                require(bool(np.all(assigned[batch]==0)), f'{name}: duplicate predictions')
                predictions[batch]=p; assigned[batch]=1
        require(bool((assigned==1).all()) and bool(np.isfinite(predictions).all()) and
                bool(((predictions>=0)&(predictions<=1)).all()),
                f'{name}: invalid or incomplete probabilities')
        archived=original_predictions(root,name,ids,labels)
        result=print_diffs(name,labels,archived,predictions)
        tn,fp,fn,tp=map(int,confusion_matrix(labels,predictions>=.5,labels=[0,1]).ravel())
        result['confusion_matrix']={'tn':tn,'fp':fp,'fn':fn,'tp':tp}
        result['checkpoint_sha256']=c['checkpoint_sha256']
        df=pd.DataFrame({'protein_id':ids,'true_label':labels,'probability_amr':predictions,
                         'prediction_at_0_5':(predictions>=.5).astype(np.int8)})
        dst=STAGE/f'{name}_frozen_unseen_policy_validation.csv'
        df.to_csv(dst,index=False,float_format='%.10g')
        result['predictions_sha256']=digest(dst)
        report['frozen_models'][name]=result
        del head
        torch.cuda.empty_cache()
    report['validation_policy_complete']=len(report['frozen_models'])==3
    require(report['validation_policy_complete'],'Policy validation incomplete')
    # Preserve only predictions + concise manifest; scratch can be regenerated
    # and is not part of the immutable protocol evidence.
    del cache
    try:
        scratch.unlink()
    except PermissionError:
        print('Warning: local scratch cannot be deleted; preserve and inspect',flush=True)
    (STAGE/'VALIDATION_POLICY_RECORD.json').write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    os.rename(STAGE,OUT)
    print('\n'+'='*74)
    print('ALL THREE FROZEN HEADS: NEW PARTITION-LOCAL POLICY VALIDATION COMPLETE')
    print('ALL THREE LORA HEADS: PRIOR VERIFIED LIVE REPLAY EVIDENCE RECHECKED')
    print('NEW INFERENCE POLICY: CANDIDATE ONLY; NOT YET LOCKED')
    print('Test/OOD: NOT OPENED | research project files: UNMODIFIED')
    print('Local output:',OUT)
    print('='*74)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test',action='store_true')
    modes.add_argument('--validate',action='store_true')
    ap.add_argument('--root',type=Path,default=Path('.'))
    args=ap.parse_args()
    if args.self_test: self_test()
    else: run(args.root)
