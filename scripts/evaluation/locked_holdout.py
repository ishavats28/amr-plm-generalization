#!/usr/bin/env python3
"""Validation-locked six-checkpoint inference and paired held-out evaluation.

--self-test: synthetic checks; no project reads.
--lock --root ... --validation-output ...: certify 3 frozen validation comparisons,
  all 6 archived replay packages, source/checkpoint hashes and protocol;
  create immutable additive inference-source lock. NEVER accesses test/OOD.
--evaluate --root ...: requires existing independently hashed inference lock;
  only THEN hashes/reads test and OOD, scores exactly 6 selected checkpoints.

No train/fit/adaptation, no checkpoint-selection, no threshold retuning.
The held-out mode cannot be tested end to end without
opening held-out datasets. Synthetic checks and validation evidence precede the inference lock.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
OUT_REL='evaluation_v2/phase2_holdout_predictions_v1'
LOCK_REL='evaluation_v2/inference_lock_v1'
BASE_SHA='71c8f971ba509e4d56ed455d2b187ca5c5197f673d08d291038dc623084e8d0c'
VALIDATE_SHA='d9adadd037a9714254bc16b9efde98d44ded5d979ada295508e762c861100110'
STRUCT_SHA='624508f5bd33a4fb6b14b32d8bf74514a05726c3132872628f32aab26fb9b727'
LOCK_PROTO_SHA='edf7610cd6b2548c8365bee5d24b8c930f5b2eac870f73b0df4433a00a830421'
LOCK_CAND_SHA='24deeeaa46ba20ad5693b4d874e4f9a29424034fd70d75b5fcbf82cdf34b842c'
EVIDENCE={
 'HypoSeq_M1_Validation_Evidence_v1.zip':'5ed58214b0b2f9b0b43a8de8ab24ea946ad6403b016227c1230cafcc521bc2d5',
 'HypoSeq_M3_LoRA_Validation_Evidence_v1.zip':'3718bc4b006f38c1c630e62afb688953be3bb8a1bf7681719e70f238c5d911c0',
 'HypoSeq_M0_LoRA_Validation_Evidence_v1.zip':'73bb490721f2178710903a3d03b0ac923322c4ebf9bdd950ccb190771e693dc7',
 'HypoSeq_M3_Frozen_Validation_Evidence_v1.zip':'1a13cb8f2ef1cac2574d7b023c8a1c7c657c1a878856eac3956e9cc10a05a43f',
 'HypoSeq_M0_Frozen_Validation_Evidence_v1.zip':'c5f2e2e9341afddd7d5ee142a8b10e94ba1b46ddc2fc88090d739cdae7759500',
}
FROZEN_ORDER=('M1','M3','M0')
CAND_ORDER=(('M1','frozen'),('M1','LoRA'),('M3','frozen'),('M3','LoRA'),('M0','frozen'),('M0','LoRA'))
PARTITIONS={'test':(1500,750,750),'ood':(642,321,321)}
THRESHOLD=0.5


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('STOP: '+reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, record):
    path=Path(path)
    tmp=path.with_name(path.name+'.partial')
    require(not path.exists() and not tmp.exists(),f'Existing file/partial: {path}')
    data=json.dumps(record,indent=2,sort_keys=True,allow_nan=False,ensure_ascii=False)+'\n'
    with tmp.open('x',encoding='utf-8') as f:
        f.write(data)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def preflight_source(require_locked_hash=True):
    import hyposeq_inference_validation_v1 as base
    import hyposeq_unseen_policy_validation_v1 as val
    if require_locked_hash:
        require(sha(Path(base.__file__))==BASE_SHA,'Base source hash mismatch')
        require(sha(Path(val.__file__))==VALIDATE_SHA,'Validation source hash mismatch')
    require(sha(HERE/'CHECKPOINT_STRUCTURE_NO_WEIGHTS.json')==STRUCT_SHA,'Checkpoint reference hash mismatch')
    return base,val


def self_test():
    # Portable copies differ only in local paths; the full locked run still checks original hashes.
    _,val=preflight_source(require_locked_hash=False)
    val.self_test()
    lens=np.array([4,100,1014,40,100,60],dtype=np.int64)
    batches=val.make_length_batches(lens)
    require(len(np.concatenate(batches))==len(lens),'Lost or duplicate protein in schedule')
    require(np.array_equal(np.concatenate(batches),np.argsort(lens,kind='stable')),
            'Deterministic schedule mismatch')
    # No dataset, checkpoint, or project read.
    print('HOLDOUT RUNNER SYNTHETIC / ORIGINAL SCHEDULER SELF-TEST: PASS')
    print('No test/OOD data opened. No project files accessed.')


def review_dev_record(root, record):
    """Predeclared frozen classifications identical; calibrated probability deltas disclosed.

    A separate within-validation partition scheme cannot exactly match the
    global train+validation historical encoder batches, so no false claim of
    probability parity or retroactive loosening of replay thresholds is made.
    """
    require(record['validation_policy_complete'] is True and record['test_ood_opened'] is False,
            'Development evidence not complete or holdout flag invalid')
    require(record['partition']=='validation' and record['n']==1500,
            'Unrecognized validation evidence partition')
    require(set(record['frozen_models'])==set(FROZEN_ORDER),'Missing frozen validation comparisons')
    require(record['base_inference_source_sha256']==BASE_SHA and
            record['original_evidence_archives_sha256']==EVIDENCE and
            record['candidates_lock_sha256']==LOCK_CAND_SHA,
            'Development validation evidence provenance differs')
    from sklearn.metrics import matthews_corrcoef
    require(record['policy']['status']=='unlocked_validation_candidate_only' and
            record['policy']['encoder_max_sequences_per_batch']==4 and
            record['policy']['encoder_max_padded_tokens_per_batch']==2048 and
            record['policy']['fixed_threshold']==0.5 and
            record['policy']['test_or_ood_inference_authorized'] is False,
            'Development policy unexpected')
    inv=load_json(root/'evaluation_v2/phase2_protocol_lock_v1/CANDIDATES_AND_EVIDENCE.json')
    for name in FROZEN_ORDER:
        r=record['frozen_models'][name]
        expected=next(c for c in inv['candidates'] if c['model']==name and c['regime']=='frozen')
        require(r['checkpoint_sha256']==expected['checkpoint_sha256'],f'{name} checkpoint mismatch')
        require(r['classification_flips']==0,f'{name}: threshold disagreements on development')
        require(r['candidate_metrics']['MCC']==r['original_metrics']['MCC'] and
                r['candidate_metrics']['F1']==r['original_metrics']['F1'],
                f'{name} development MCC/F1 changed')
        require(np.isfinite(r['maximum_absolute_probability_difference']) and
                r['maximum_absolute_probability_difference']>=0,
                f'{name} probability deltas invalid')
        print(f'{name}: original vs partition-local MCC/F1 SAME, flips=0; '
              f'mean |Δp|={r["mean_absolute_probability_difference"]:.9f}',flush=True)
    return inv


def attest_archives(root):
    for name, expected in EVIDENCE.items():
        f=root/'evaluation_v2'/name
        require(f.is_file() and sha(f)==expected,f'Archive missing/changed: {name}')
        with zipfile.ZipFile(f) as z:
            require(z.testzip() is None,f'Archive corruption: {name}')
            manifest=json.loads(z.read('EVIDENCE_MANIFEST.json'))
            for k,v in manifest.items():
                require(hashlib.sha256(z.read(k)).hexdigest()==v,f'Archive member mismatch: {name}/{k}')
    print('ALL FIVE PRIOR REPLAY EVIDENCE ARCHIVES VERIFIED',flush=True)


def lock(root,development):
    base,val=preflight_source()
    root=Path(root)
    inv,proto=base.verified_preflight(root)
    base.verify_checkpoint_structures(root,inv)
    base.runtime_guard(proto)
    attest_archives(root)
    record_path=Path(development)/'VALIDATION_POLICY_RECORD.json'
    require(record_path.is_file(),f'Validation evidence missing: {record_path}')
    record=load_json(record_path)
    review_dev_record(root,record)
    # Metadata-only check that both eventual input paths are certified by the
    # frozen split manifests. This does NOT open/read either held-out file.
    split_manifest=load_json(root/'metadata/v2_split_manifest.json')
    split_audit=load_json(root/'metadata/v2_split_audit.json')
    processed=[rel for rel in split_manifest['output_hashes']
               if rel.startswith('data/processed/') and rel.endswith('.tsv.gz')]
    for part in PARTITIONS:
        matches=[rel for rel in processed if Path(rel).stem.split('.')[0]==part]
        require(len(matches)==1,f'Missing/ambiguous certified {part} file name in metadata: {processed}')
        rel=matches[0]
        require(split_manifest['output_hashes'][rel]==split_audit['verified_output_hashes'][rel],
                f'{part} metadata certification mismatch')
        print(f'HELDOUT FILE METADATA NAME VERIFIED (NO FILE READ): {rel}',flush=True)
    for model,r in record['frozen_models'].items():
        csv=Path(development)/f'{model}_frozen_unseen_policy_validation.csv'
        require(csv.is_file() and sha(csv)==r['predictions_sha256'],
                f'Archived development prediction hash mismatch {model}')
    lock_path=root/LOCK_REL
    require(not lock_path.exists(),f'Inference lock already exists: {lock_path}. NEVER overwrite it.')
    require(not (root/OUT_REL).exists(),'Holdout result directory already exists; inspect it')
    # Only code, validation evidence, metadata, and checkpoint structure read so far.
    # Held-out filenames, bytes, sample labels or sequences are never opened here.
    lock_path.mkdir(parents=True,exist_ok=False)
    for name in ('hyposeq_locked_holdout_v1.py','hyposeq_unseen_policy_validation_v1.py',
                 'hyposeq_inference_validation_v1.py','CHECKPOINT_STRUCTURE_NO_WEIGHTS.json'):
        shutil.copyfile(HERE/name,lock_path/name)
        require(sha(HERE/name)==sha(lock_path/name),f'Lock source copy failed: {name}')
    for name in ['VALIDATION_POLICY_RECORD.json']+[f'{n}_frozen_unseen_policy_validation.csv' for n in FROZEN_ORDER]:
        src=Path(development)/name
        dest=lock_path/name
        shutil.copyfile(src,dest)
        require(sha(src)==sha(dest),f'Validation evidence copy failed: {name}')
    policy={
        'schema':'hyposeq_v2_independent_inference_lock_v1',
        'status':'locked_source_and_inference_policy',
        'protocol_sha256':LOCK_PROTO_SHA,
        'candidates_sha256':LOCK_CAND_SHA,
        'source':{'python':sys.version.split()[0],'torch':proto['software']['torch'],
                  'transformers':proto['software']['transformers'],
                  'peft':proto['software']['peft'],'accelerate':proto['software']['accelerate']},
        'encoder_backbone_revision':base.ESM_REV,
        'frozen_encoder_batching':'stable_ascending_sequence_length_within_each_partition',
        'frozen_batch_max_sequences':4,
        'frozen_batch_max_padded_tokens':2048,
        'frozen_encoder_dtype':'bfloat16',
        'frozen_residues_storage_dtype':'float16',
        'frozen_m0_pooling':'unrounded BF16 residue vectors cast FP32; mean FP32; store FP16; head FP32',
        'frozen_m1_m3_features':'original residue sequence embeddings stored FP16; head FP32',
        'lora_encoder_batching':'stable_ascending_sequence_length_within_partition_in_pairs',
        'lora_batch_size':2,
        'lora_encoder_head_dtype':'bfloat16_autocast_with_original_live_head',
        'model_order':[list(x) for x in CAND_ORDER],
        'partition_order':['test','ood'],
        'threshold':0.5,
        'no_truncation':True,
        'selection':'fixed_checkpoint_by_prior_phase2_protocol',
        'development_probabilities':'small_archived_vs_partition_local_differences_documented; no_0.5_flips',
        'evaluation_input_order':'unique_protein_id_stable_sort',
        'post_hoc_selection_or_recalibration':False,
        'scope':'test_and_ood_inference_only',
        'analysis_protocol':'existing Phase2 PROTOCOL.md fixed; this lock does not change it',
    }
    atomic_json(lock_path/'INFERENCE_POLICY.json',policy)
    files={f.name:sha(f) for f in lock_path.iterdir() if f.is_file()}
    manifest={'schema':'hyposeq_inference_code_lock_manifest_v1',
              'status':'locked','policy':policy['schema'],'files':files,
              'test_ood_opened_at_lock':False,
              'validation_record_sha256':sha(record_path),
              'preexisting_replay_evidence_sha256':EVIDENCE}
    atomic_json(lock_path/'INFERENCE_LOCK_MANIFEST.json',manifest)
    print('\nINFERENCE SOURCE/DEVELOPMENT EVIDENCE LOCK: CREATED',flush=True)
    print('Location:',lock_path,flush=True)
    print('Lock manifest SHA256:',sha(lock_path/'INFERENCE_LOCK_MANIFEST.json'),flush=True)
    print('No held-out files opened. Do not change this lock.',flush=True)


def verify_lock(root):
    root=Path(root)
    where=root/LOCK_REL
    manifest_path=where/'INFERENCE_LOCK_MANIFEST.json'
    require(manifest_path.is_file(),'Inference lock absent. DO NOT access holdouts')
    manifest=load_json(manifest_path)
    require(manifest['schema']=='hyposeq_inference_code_lock_manifest_v1' and
            manifest['status']=='locked' and manifest['test_ood_opened_at_lock'] is False,
            'Inference lock status not approved')
    files=manifest['files']
    require(set(files)=={p.name for p in where.iterdir() if p.is_file()}-{'INFERENCE_LOCK_MANIFEST.json'},
            'Inference lock file inventory changed')
    for n, expected in files.items():
        require(sha(where/n)==expected,f'Inference lock changed: {n}')
    require(files['hyposeq_locked_holdout_v1.py']==sha(Path(__file__)),
            'Running code differs from independently locked source')
    require(files['hyposeq_inference_validation_v1.py']==BASE_SHA and
            files['hyposeq_unseen_policy_validation_v1.py']==VALIDATE_SHA and
            files['CHECKPOINT_STRUCTURE_NO_WEIGHTS.json']==STRUCT_SHA,
            'Locked original interfaces do not match pinned hashes')
    policy=load_json(where/'INFERENCE_POLICY.json')
    require(policy['status']=='locked_source_and_inference_policy' and
            policy['threshold']==.5 and policy['no_truncation'] is True and
            policy['model_order']==[list(x) for x in CAND_ORDER] and
            policy['partition_order']==['test','ood'],
            'Locked inference policy modified')
    require(manifest['validation_record_sha256']==files['VALIDATION_POLICY_RECORD.json'],
            'Lock evidence record hash mismatch')
    review_dev_record(root,load_json(where/'VALIDATION_POLICY_RECORD.json'))
    print('IMMUTABLE INFERENCE SOURCE LOCK: PASS',flush=True)
    return policy


def certified_partition_metadata(root, part):
    """Manifest-only lookups, no holdout file accessed during locking."""
    from common import verify_attestation
    report,manifest=verify_attestation(root)
    prefixes=[p for p in manifest['output_hashes'] if p.startswith('data/processed/') and p.endswith('.tsv.gz')]
    # Match exact basename test or OOD without inferring files outside certified manifest.
    matches=[p for p in prefixes if Path(p).stem.split('.')[0] == part]
    require(len(matches)==1,f'Exactly one certified {part} TSV.GZ expected; names found {prefixes}')
    rel=matches[0]
    require(manifest['output_hashes'][rel]==report['verified_output_hashes'][rel],
            f'{part}: certification manifests differ')
    return rel,manifest['output_hashes'][rel]


def open_partition(root,part):
    import pandas as pd
    require(part in PARTITIONS,'Unknown holdout partition')
    rel,digest=certified_partition_metadata(root,part)
    path=root/rel
    require(path.is_file() and sha(path)==digest,f'{part}: heldout file differs from certified bytes')
    df=pd.read_csv(path,sep='\t',dtype={'protein_id':str,'sequence':str})
    needed={'protein_id','sequence','length','label','split','homology_component'}
    require(needed.issubset(df.columns),f'{part}: required columns missing')
    n,n0,n1=PARTITIONS[part]
    require(len(df)==n and df['protein_id'].notna().all() and df['protein_id'].is_unique,
            f'{part}: ID/count mismatch')
    require(df['sequence'].notna().all() and df['sequence'].str.len().eq(df['length']).all(),
            f'{part}: sequence lengths invalid')
    require(df['sequence'].map(lambda s: bool(s) and len(s)<=1014 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY')).all(),
            f'{part}: noncanonical or overlength sequence')
    require(df['split'].eq(part).all(),f'{part}: split membership differs')
    require(df['label'].isin((0,1)).all() and int(df['label'].eq(0).sum())==n0 and
            int(df['label'].eq(1).sum())==n1,f'{part}: labels/counts differ')
    require(df['homology_component'].notna().all(),f'{part}: homology component missing')
    df=df.sort_values('protein_id',kind='stable').reset_index(drop=True)
    print(f'{part.upper()}: CERTIFIED FILE SHA256 PASS; {n} IDs; ({n0},{n1}) labels',flush=True)
    return df,rel,digest


def scored_metrics(y,probs):
    from sklearn.metrics import (matthews_corrcoef,f1_score,roc_auc_score,
              average_precision_score,confusion_matrix,recall_score,
              balanced_accuracy_score,precision_score,accuracy_score)
    y=np.asarray(y,dtype=np.int64);p=np.asarray(probs,dtype=np.float64)
    require(len(y)==len(p) and bool(np.isfinite(p).all()) and bool(((p>=0)&(p<=1)).all()),
            'Incomplete or nonfinite probabilities')
    pred=(p>=THRESHOLD).astype(np.int64)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return {'mcc':float(matthews_corrcoef(y,pred)), 'f1':float(f1_score(y,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'brier':float(np.mean((p-y)**2)),
            'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'sensitivity_recall':float(recall_score(y,pred)),
            'specificity':float(tn/(tn+fp)),
            'prevalence_positive':float(y.mean()),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp}


def save_predictions(root,part,df,rel,digest,model,regime,candidate,probabilities,source_sha):
    import pandas as pd
    results=root/OUT_REL
    label=f'{part}_{model}_{regime}'
    where=results/label
    partial=results/(label+'.partial')
    require(not where.exists() and not partial.exists(),f'Existing result/partial: {label}')
    partial.mkdir(parents=True,exist_ok=False)
    metrics=scored_metrics(df['label'].to_numpy(dtype=np.int64),probabilities)
    dst=partial/'predictions.csv'
    tmpdf=pd.DataFrame({'protein_id':df['protein_id'].astype(str),
             'true_label':df['label'].astype(np.int64),
             'homology_component':df['homology_component'].astype(str),
             'probability_amr':probabilities,
             'prediction_at_0_5':(probabilities>=.5).astype(np.int8)})
    tmpdf.to_csv(dst,index=False,float_format='%.10g')
    record={'schema':'hyposeq_v2_locked_holdout_prediction_v1','model':model,'regime':regime,
        'partition':part,'input_relative_path':rel,'input_sha256':digest,'n':len(df),
        'checkpoint_sha256':candidate['checkpoint_sha256'],
        'selected_seed':candidate['seed'],'selected_epoch':candidate['selected_epoch'],
        'inference_runner_sha256':source_sha,'phase2_protocol_sha256':LOCK_PROTO_SHA,
        'candidate_inventory_sha256':LOCK_CAND_SHA,
        'inference_lock_manifest_sha256':sha(root/LOCK_REL/'INFERENCE_LOCK_MANIFEST.json'),
        'threshold':0.5,'metrics':metrics,'predictions_csv_sha256':sha(dst),
        'label_order':'protein_id_ascending','completed':True}
    atomic_json(partial/'RESULT.json',record)
    os.replace(partial,where)
    print('RESULT:',label,'MCC=',f'{metrics["mcc"]:.6f}',
          'AUROC=',f'{metrics["roc_auc"]:.6f}','SHA256=',record['predictions_csv_sha256'],flush=True)
    return record


def verify_previous(root,part,df,rel,digest,model,regime,candidate):
    """If already committed, skip only when complete and provenance matches lock."""
    label=f'{part}_{model}_{regime}'
    where=root/OUT_REL/label
    partial=root/OUT_REL/(label+'.partial')
    require(not partial.exists(),f'Partial output exists, inspect and preserve: {partial}')
    if not where.exists():return False
    rec=load_json(where/'RESULT.json')
    require(rec['schema']=='hyposeq_v2_locked_holdout_prediction_v1' and
            rec['completed'] is True and rec['model']==model and rec['regime']==regime and
            rec['partition']==part and rec['n']==len(df) and
            rec['input_relative_path']==rel and rec['input_sha256']==digest and
            rec['checkpoint_sha256']==candidate['checkpoint_sha256'] and
            rec['inference_runner_sha256']==sha(Path(__file__)) and
            rec['inference_lock_manifest_sha256']==sha(root/LOCK_REL/'INFERENCE_LOCK_MANIFEST.json') and
            rec['threshold']==.5 and rec['predictions_csv_sha256']==sha(where/'predictions.csv'),
            f'Existing result differs from lock or input: {label}')
    print('SKIP EXISTING VERIFIED:',label,flush=True)
    return True


def frozen_shared_features(root,df,partition):
    """One ESM-2 frozen backbone forward pass per protein, three original FP32 heads."""
    import torch
    import hyposeq_unseen_policy_validation_v1 as val
    from contextlib import suppress
    ext_path=root/'training_v2/M1_esm2_multiscale_cnn/extract.py'
    require(sha(ext_path)==val.M1_EXTRACT_SHA,'Original M1 extractor changed')
    ext=val.import_file('hyposeq_original_M1_for_heldout',ext_path)
    cfg=load_json(root/'training_v2/M1_esm2_multiscale_cnn/config.json')
    lengths=df['length'].to_numpy(dtype=np.int64)
    seq=df['sequence'].astype(str).tolist()
    batches=val.make_length_batches(lengths)
    order=np.argsort(lengths,kind='stable')
    cursor=0
    for batch in batches:
        require(np.array_equal(batch,np.asarray(ext.plan_batch(order,cursor,lengths,4,2048))),
                'Frozen batch planner differs from original')
        cursor+=len(batch)
    require(cursor==len(df),'Bad complete frozen batches')
    scratch=Path(tempfile.gettempdir())/f'.hyposeq_{partition}_locked_frozen_features.partial.npy'
    require(not scratch.exists(),f'Existing frozen feature scratch; inspect: {scratch}')
    offset=np.zeros(len(df)+1,dtype=np.int64)
    np.cumsum(lengths,out=offset[1:])
    cache=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float16,
                shape=(int(offset[-1]),1280))
    pooled=np.empty((len(df),1280),dtype=np.float16)
    print(f'{partition}: frozen encoder {len(batches)} batches; scratch {scratch.stat().st_size/2**30:.2f} GiB',flush=True)
    tokenizer,encoder=ext.load_encoder(cfg)
    for i,b in enumerate(batches,1):
        v,p=val.encode_original_precision_with_pooled([seq[int(k)] for k in b],tokenizer,encoder)
        for idx,emb,pool in zip(b,v,p):
            require(emb.shape==(int(lengths[idx]),1280),'Feature length mismatch')
            cache[offset[idx]:offset[idx+1]]=emb
            pooled[idx]=pool
        if i%75==0 or i==len(batches):
            print(f'  {partition}: frozen ESM2 {i}/{len(batches)}',flush=True)
    cache.flush();del cache,encoder,tokenizer
    torch.cuda.empty_cache()
    return scratch,offset,pooled


def frozen_score(root,model,candidate,df,scratch,offset,pooled):
    import torch
    import hyposeq_inference_validation_v1 as base
    cfg=load_json(root/base.HOME[model]/'config.json')
    head=base.make_head(root,model,cfg).to('cuda:0')
    saved=torch.load(root/candidate['checkpoint_relative_path'],map_location='cpu',weights_only=True)
    require(saved['epoch']==candidate['selected_epoch'],'Frozen selected epoch mismatch')
    head.load_state_dict(saved['model'],strict=True)
    head.eval();del saved
    lengths=df['length'].to_numpy(dtype=np.int64)
    if model=='M0':
        bs=int(cfg['batch_size'])
        batches=[np.arange(i,min(i+bs,len(df)),dtype=np.int64)
                 for i in range(0,len(df),bs)]
    else:
        extmod=__import__('hyposeq_unseen_policy_validation_v1')
        folder=root/base.HOME[model]
        sys.path.insert(0,str(folder))
        for key in ('model','cache'):
            sys.modules.pop(key,None)
        trainer=extmod.import_file('hyposeq_final_original_'+model+'_trainer',
                                    folder/'train.py')
        batches=trainer.make_batches(np.arange(len(df),dtype=np.int64),lengths,
                                    int(cfg['batch_size']),int(cfg['max_padded_residues_per_batch']),
                                    shuffle=False,rng=torch.Generator(device='cpu').manual_seed(42))
    mem=np.load(scratch,mmap_mode='r',allow_pickle=False)
    p=np.full(len(df),np.nan,dtype=np.float64)
    seen=np.zeros(len(df),dtype=np.int8)
    with torch.inference_mode():
        for batch in batches:
            if model=='M0':
                x=torch.from_numpy(pooled[batch].astype(np.float32)).to('cuda:0')
                logits=head(x)
            else:
                pad=np.zeros((len(batch),int(lengths[batch].max()),1280),dtype=np.float32)
                for i,idx in enumerate(batch):
                    pad[i,:int(lengths[idx])]=mem[offset[idx]:offset[idx+1]]
                x=torch.from_numpy(pad).to('cuda:0')
                lenx=torch.from_numpy(lengths[batch].astype(np.int64)).to('cuda:0')
                logits=head(x,lenx)
            require(logits.shape==(len(batch),) and bool(torch.isfinite(logits).all()),
                    'Frozen classifier forward failed')
            require(bool(np.all(seen[batch]==0)),'Duplicate frozen predictions')
            p[batch]=torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
            seen[batch]=1
    require(bool((seen==1).all()) and bool(np.isfinite(p).all()),
            'Frozen predictions incomplete')
    del mem,head
    torch.cuda.empty_cache()
    return p


def lora_score(root,inv,proto,model,df):
    import torch
    import hyposeq_inference_validation_v1 as base
    device=torch.device('cuda:0')
    tokenizer,encoder,head=base.load_checkpoint_and_models(root,inv,proto,model,'LoRA',device)
    lengths=df['length'].to_numpy(dtype=np.int64)
    order=np.argsort(lengths,kind='stable')
    p=np.full(len(df),np.nan,dtype=np.float64)
    seen=np.zeros(len(df),dtype=np.int8)
    with torch.inference_mode():
        for start in range(0,len(df),2):
            batch=order[start:start+2]
            ids,attention,mask,L=base.tokenize_residues(tokenizer,
                   df.iloc[batch]['sequence'].astype(str).tolist(),device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=encoder(input_ids=ids,attention_mask=attention).last_hidden_state
                features=base.residue_features(hidden,mask,model)
                logits=head(features) if model=='M0' else head(features,L)
            require(logits.shape==(len(batch),) and bool(torch.isfinite(logits).all()),
                    'LoRA nonfinite/malformed classifier output')
            require(bool(np.all(seen[batch]==0)),'Duplicate LoRA prediction indices')
            p[batch]=torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
            seen[batch]=1
            if (start+len(batch))%300==0 or start+len(batch)==len(df):
                print(f'  {model} LoRA: {start+len(batch)}/{len(df)}',flush=True)
    require(bool((seen==1).all()) and bool(np.isfinite(p).all()),
            'Incomplete LoRA predictions')
    del encoder,head,tokenizer
    torch.cuda.empty_cache()
    return p


def evaluate(root):
    import torch
    root=Path(root)
    # *** Access boundary: no heldout data may be opened BEFORE these pass. ***
    verify_lock(root)
    base,val=preflight_source()
    inv,proto=base.verified_preflight(root)
    base.verify_checkpoint_structures(root,inv)
    base.runtime_guard(proto)
    attest_archives(root)
    require(sha(root/'evaluation_v2/phase2_protocol_lock_v1/PROTOCOL.md')==LOCK_PROTO_SHA,
            'locked held-out protocol checksum mismatch')
    require(sha(root/'evaluation_v2/phase2_protocol_lock_v1/CANDIDATES_AND_EVIDENCE.json')==LOCK_CAND_SHA,
            'Candidate manifest changed')
    sys.path.insert(0,str(root/'training_v2'))
    from common import verify_attestation
    verify_attestation(root)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    candidates={(c['model'],c['regime']):c for c in inv['candidates']}
    results_root=root/OUT_REL
    results_root.mkdir(parents=True,exist_ok=True)
    for part in ('test','ood'):
        df,rel,hash_=open_partition(root,part) # ONLY HERE are heldouts opened.
        frozen_pending=[m for m in FROZEN_ORDER if not verify_previous(root,part,df,rel,hash_,m,'frozen',candidates[(m,'frozen')])]
        if frozen_pending:
            scratch,offset,pooled=frozen_shared_features(root,df,part)
            try:
                for model in FROZEN_ORDER:
                    if model not in frozen_pending:continue
                    probs=frozen_score(root,model,candidates[(model,'frozen')],df,scratch,offset,pooled)
                    save_predictions(root,part,df,rel,hash_,model,'frozen',candidates[(model,'frozen')],
                                     probs,sha(Path(__file__)))
            finally:
                # scratch only removed on successful scoring of all pending heads.
                # On errors leave scratch intact for postmortem; do not silently reuse.
                if all((results_root/f'{part}_{m}_frozen').exists() for m in frozen_pending):
                    try:scratch.unlink()
                    except OSError:pass
        # Independent LoRA checkpoints. Never load multiple ESM-2 650M models simultaneously.
        for model in FROZEN_ORDER:
            cand=candidates[(model,'LoRA')]
            if verify_previous(root,part,df,rel,hash_,model,'LoRA',cand):continue
            probs=lora_score(root,inv,proto,model,df)
            save_predictions(root,part,df,rel,hash_,model,'LoRA',cand,probs,sha(Path(__file__)))
    finished=[]
    for part in ('test','ood'):
        for model,regime in CAND_ORDER:
            p=results_root/f'{part}_{model}_{regime}'
            require((p/'RESULT.json').is_file(),'Missing result '+str(p))
            finished.append(load_json(p/'RESULT.json'))
    # Preserve all row-level predictions plus individual result records. Statistical
    # bootstrap and block analysis are a separate, protocol-predefined next stage.
    manifest={'schema':'hyposeq_v2_phase2_12_confirmatory_evaluations_v1',
              'status':'inference_complete','inference_lock_manifest_sha256':
                sha(root/LOCK_REL/'INFERENCE_LOCK_MANIFEST.json'),
              'results':[{k:r[k] for k in ('partition','model','regime','predictions_csv_sha256',
                            'input_sha256','checkpoint_sha256','metrics')} for r in finished]}
    summary=results_root/'FINAL_INFERENCE_MANIFEST.json'
    require(not summary.exists(),'Summary manifest already exists; inspect')
    atomic_json(summary,manifest)
    print('\n'+('='*72))
    print('SIX CHECKPOINTS x TWO SEALED PARTITIONS: ALL 12 INFERENCES COMPLETE')
    print('Predictions:',results_root)
    print('Summary SHA256:',sha(summary))
    print('Next: locked bootstrap and OOD biological-block statistics')
    print('='*72)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    g=a.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--lock',action='store_true')
    g.add_argument('--evaluate',action='store_true')
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--validation-output',type=Path,default=Path('validation_policy_output'))
    args=a.parse_args()
    if args.self_test:self_test()
    elif args.lock:lock(args.root,args.validation_output)
    else:evaluate(args.root)

if __name__=='__main__':main()
