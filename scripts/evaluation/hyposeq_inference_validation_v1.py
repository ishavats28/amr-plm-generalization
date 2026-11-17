#!/usr/bin/env python3
"""Validation-only inference replay interface for the selected checkpoints.

Modes:
  --self-test : synthetic CPU checks, never opens research project
  --preflight : verify locked source/protocol/checkpoint metadata, never reads datasets
  --validate --model M1 --regime frozen|LoRA : real 1,500-protein validation replay

Intentionally NO test/OOD input interface or generic FASTA scoring interface.
Never writes to the research root. Real validation predictions go to a separate local output directory.
Requires pinned runtime environment for --validate; no training/optimizer operations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

LOCK_PROTOCOL_SHA = 'edf7610cd6b2548c8365bee5d24b8c930f5b2eac870f73b0df4433a00a830421'
LOCK_CANDIDATES_SHA = '24deeeaa46ba20ad5693b4d874e4f9a29424034fd70d75b5fcbf82cdf34b842c'
FT_PROTOCOL_SHA = '7ee4e5f963403568e0adceebac98afdb029709b92b80ff3479177309a6459cad'
AUDIT_SHA = 'f587a237fdf4a48b1beaa559cdd56e2461cf46b600b1038e3c5969eb29a6e8ac'
ESM_REV = '08e4846e537177426273712802403f7ba8261b6c'
SOURCE_HASHES = {
 'finetuning_v2/M0_lora/train.py':'f034fd39c4e70b674520f796166032e5b2b7977f8f908b515596ded7d80474f0',
 'finetuning_v2/M1_lora/train.py':'00b63261b791879d5c6b9f633304ea87a4d2a16b8a76a460e01d1a7e1af7b4ca',
 'finetuning_v2/M3_lora/train.py':'938405c6c01efa4e4d9df9698b0ea123807a19ce1c07aef81f6477277f4b30f2',
 'training_v2/common.py':'86362aba1f2c4b42a15463dd573e9c257b3289877dbe4024597c28a758e7332f',
 'training_v2/M0_esm2_mlp/train.py':'916e0681c7c23490d5bd2db60fc9a1d62c64adf0cd6b070ccdb5ac4ff448bd19',
 'training_v2/M0_esm2_mlp/config.json':'cd5f35354f0c367af23fc9e06a6eb404edfa1900405620e5368870b8e2338b11',
 'training_v2/M1_esm2_multiscale_cnn/train.py':'4ec73eaad1692b1f173bf7f4c37f0d4ef323937e2cadea9428386cd8a04150b9',
 'training_v2/M1_esm2_multiscale_cnn/model.py':'458c81e4e9b278a077b0ee0d32084314fc8f7cf810ff3dc3a4684d73fbb08533',
 'training_v2/M1_esm2_multiscale_cnn/config.json':'d2af557f768182dd207ffdafa58e96054be21212e77e155d867919d3f0f047b2',
 'training_v2/M3_esm2_bilstm_transformer/train.py':'72cb2667f5d072f9885bb97f2c89ddf135221d02bc4600dd756ab2e20c7b6fb9',
 'training_v2/M3_esm2_bilstm_transformer/model.py':'52855ceca628162171d8b0a4e04cf536c923c8a896448f091be23ffee736c46d',
 'training_v2/M3_esm2_bilstm_transformer/cache.py':'c820468dc4e67036eb75c307457a26442d4a4611c6dce6d2c035bc03e2c7d0ff',
 'training_v2/M3_esm2_bilstm_transformer/config.json':'b0a44accf66fd293b7705ed279e28f34599e9129300bda98ebca4d32d72abd91',
}
HOME = {
 'M0':'training_v2/M0_esm2_mlp',
 'M1':'training_v2/M1_esm2_multiscale_cnn',
 'M3':'training_v2/M3_esm2_bilstm_transformer',
}
# Strict, specified BEFORE validation replay. Actual differences are always printed.
# Exact frozen-cache/live-BF16 prediction parity is *not* assumed.
MCC_TOL = 0.000001
AUC_AP_TOL = 0.000020
BRIER_TOL = 0.000050
PREDICTION_THRESHOLD = 0.5


def require(ok, message):
    if not bool(ok):
        raise RuntimeError('STOP: ' + message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def json_load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def import_file(name, path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    require(spec is not None and spec.loader is not None, f'Cannot import {path}')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verified_preflight(root, include_checkpoint_hashes=True):
    root=Path(root)
    require(root.is_dir(),f'Project root not found: {root}')
    lock=root/'evaluation_v2/phase2_protocol_lock_v1'
    require(sha256(lock/'PROTOCOL.md')==LOCK_PROTOCOL_SHA,'Locked evaluation protocol changed')
    require(sha256(lock/'CANDIDATES_AND_EVIDENCE.json')==LOCK_CANDIDATES_SHA,
            'Candidate inventory changed')
    lm=json_load(lock/'LOCK_MANIFEST.json')
    require(lm.get('status')=='locked_checkpoint_and_analysis_plan_inference_code_pending',
            'Unexpected lock status')
    require(lm['files']=={'PROTOCOL.md':LOCK_PROTOCOL_SHA,
                          'CANDIDATES_AND_EVIDENCE.json':LOCK_CANDIDATES_SHA},
            'Lock manifest files differ')
    require(lm.get('test_ood_data_files_read') is False,'Historical holdout flag changed')
    require(sha256(root/'finetuning_v2/protocol_v1.json')==FT_PROTOCOL_SHA,
            'Original LoRA protocol hash changed')
    require(sha256(root/'metadata/v2_split_audit.json')==AUDIT_SHA,
            'certified split-audit checksum changed')
    protocol=json_load(root/'finetuning_v2/protocol_v1.json')
    require(protocol.get('status')=='locked' and protocol['backbone']['commit']==ESM_REV,
            'Fine-tuning protocol identity changed')
    require(protocol['training']['no_sequence_truncation'] is True,
            'Protocol truncation altered')
    require(protocol['training']['classification_threshold']==0.5,
            'Protocol threshold changed')
    require(sha256(root/'benchmark_v2_freeze/PACKAGE_MANIFEST.json')==protocol['baseline_freeze_sha256'],
            'Frozen baseline freeze changed')
    require(sha256(root/'benchmark_v2_validation_addendum/summary/FINAL_MANIFEST.json')==protocol['validation_addendum_sha256'],
            'Validation addendum changed')
    for name, digest in SOURCE_HASHES.items():
        require(sha256(root/name)==digest,f'Source changed: {name}')
    inv=json_load(lock/'CANDIDATES_AND_EVIDENCE.json')
    require(inv['schema']=='hyposeq_v2_phase2_candidates_and_evidence_v1' and
            inv['backbone_revision']==ESM_REV and
            inv['dataset_version']=='V2' and inv['primary_model']=='M1' and
            inv['primary_endpoint']=='test_MCC_at_0.5_delta_LoRA_minus_frozen',
            'Unexpected locked inventory identity')
    candidates=inv['candidates']
    require(len(candidates)==6,'Expected six locked candidates')
    require([(c['model'],c['regime'],c['seed'],c['selected_epoch']) for c in candidates]==[
            ('M1','frozen',42,1),('M1','LoRA',62,2),
            ('M3','frozen',42,4),('M3','LoRA',52,6),
            ('M0','frozen',42,6),('M0','LoRA',62,3)], 'Candidate roster changed')
    required_names=set()
    for c in candidates:
        relative=Path(c['checkpoint_relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe candidate checkpoint path')
        required_names.add(c['checkpoint_relative_path'])
        if include_checkpoint_hashes:
            require(sha256(root/relative)==c['checkpoint_sha256'],
                    f"Checkpoint hash mismatch: {c['model']} {c['regime']}")
        if c['regime']=='LoRA':
            summary_rel=Path(c['training_summary_relative_path'])
            require(sha256(root/summary_rel)==c['training_summary_sha256'],
                    f"Training summary mismatch {c['model']}")
            summary=json_load(root/summary_rel)
            require(summary['status']=='complete' and
                    summary['best_model_sha256']==c['checkpoint_sha256'] and
                    summary['seed']==c['seed'] and summary['best_epoch']==c['selected_epoch'] and
                    summary['test_ood_accessed'] is False,
                    f"LoRA candidate summary inconsistent {c['model']}")
    require(len(required_names)==6,'Duplicate locked checkpoint path')
    print('LOCK + SOURCE PROVENANCE: PASS')
    print('SIX CHECKPOINTS: '+('SHA256 VERIFIED' if include_checkpoint_hashes else 'metadata only'))
    print('No test/OOD files opened by this script.')
    return inv, protocol


def verify_checkpoint_structures(root, inv):
    """Compare actual six weights-only checkpoint schemas with CPU-safe inventory."""
    import torch
    inventory_path=Path(__file__).parent/'CHECKPOINT_STRUCTURE_NO_WEIGHTS.json'
    require(inventory_path.is_file(),'Checkpoint structure reference is missing')
    reference=json_load(inventory_path)
    for c in inv['candidates']:
        tag=f"{c['model']}_{c['regime']}"
        require(tag in reference,f'Missing structural reference {tag}')
        r=reference[tag]
        require(r['relative_path']==c['checkpoint_relative_path'] and
                r['sha256']==c['checkpoint_sha256'],f'Checkpoint reference identity mismatch {tag}')
        ck=torch.load(root/c['checkpoint_relative_path'],map_location='cpu',weights_only=True)
        require(set(ck)==set(r['top_level_keys']),f'{tag} checkpoint top-level keys changed')
        for k,value in r['scalar_metadata'].items():
            require(ck[k]==value,f'{tag} scalar checkpoint metadata changed: {k}')
        for state_name,items in r['tensor_states'].items():
            state=ck[state_name]
            require(len(items)==len(state) and set(state)=={item['key'] for item in items},
                    f'{tag} tensor state keys changed: {state_name}')
            for item in items:
                tensor=state[item['key']]
                require(list(tensor.shape)==item['shape'] and str(tensor.dtype)==item['dtype'],
                        f"{tag} tensor shape or dtype mismatch: {item['key']}")
        del ck
        print(f'{tag}: TENSOR STRUCTURE PASS',flush=True)
    print('SIX SELECTED CHECKPOINT TENSOR SCHEMAS: PASS',flush=True)


def make_head(root, model, cfg):
    """Instantiate EXACT original architecture classes, after SHA256 verification."""
    import torch
    root=Path(root)
    if model=='M0':
        sys.path.insert(0,str(root/'training_v2'))
        mod=import_file('hyposeq_verified_m0_frozen_head',root/HOME['M0']/'train.py')
        head=mod.M0Head(tuple(cfg['head_widths']),float(cfg['dropout']))
    elif model=='M1':
        mod=import_file('hyposeq_verified_m1_model',root/HOME['M1']/'model.py')
        head=mod.MultiScaleResidueCNN(input_dim=int(cfg['hidden_size']),
                kernels=tuple(cfg['cnn_kernels']),channels=int(cfg['cnn_channels_per_kernel']),
                widths=tuple(cfg['head_widths']),dropout=float(cfg['dropout']))
    elif model=='M3':
        mod=import_file('hyposeq_verified_m3_model',root/HOME['M3']/'model.py')
        head=mod.ResidueBiLSTMTransformer(input_dim=int(cfg['hidden_size']),
                model_dim=int(cfg['transformer_model_dim']),heads=int(cfg['transformer_heads']),
                transformer_layers=int(cfg['transformer_layers']),
                lstm_hidden_per_direction=int(cfg['lstm_hidden_per_direction']),
                lstm_layers=int(cfg['lstm_layers']),
                lstm_output_dropout=float(cfg['lstm_output_dropout']),
                feedforward_dim=int(cfg['transformer_feedforward_dim']),
                head_widths=tuple(cfg['head_widths']),dropout=float(cfg['dropout']),
                transformer_dropout=float(cfg['transformer_dropout']))
    else:
        raise RuntimeError('Unknown model')
    expected={'M0':344449,'M1':2671361,'M3':1927042}[model]
    require(sum(p.numel() for p in head.parameters())==expected,
            f'{model} original head parameter count mismatch')
    return head


def tokenize_residues(tokenizer, sequences, device):
    """Verified original get_tokens rules, with no labels or split access."""
    import torch
    from torch.nn.utils.rnn import pad_sequence
    require(all(isinstance(s,str) and 1<=len(s)<=1014 for s in sequences),
            'Invalid protein sequence length')
    p=tokenizer(list(sequences),return_tensors='pt',padding=True,
                truncation=False,return_special_tokens_mask=True)
    ids=p['input_ids'].to(device)
    attention=p['attention_mask'].to(device)
    mask=attention.bool() & ~p['special_tokens_mask'].to(device).bool()
    lengths=torch.tensor([len(s) for s in sequences],device=device,dtype=torch.long)
    require(ids.ndim==2 and ids.shape[1]<=1024,'ESM2 token limit exceeded')
    require(torch.equal(mask.sum(dim=1),lengths),'Residue tokenization mismatch')
    return ids,attention,mask,lengths


def residue_features(hidden, mask, model):
    """Identical to the original M0 versus M1/M3 feature selection."""
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if model=='M0':
        valid=mask.unsqueeze(-1)
        n=(hidden.float()*valid).sum(dim=1)
        d=valid.sum(dim=1)
        require(bool((d>0).all()),'M0 empty residue mask')
        return n/d
    return pad_sequence([hidden[i][mask[i]].float() for i in range(len(mask))],
                        batch_first=True,padding_value=0.)


def self_test():
    import torch
    from torch import nn
    root=Path(__file__).parent
    # A fake tokenizer is adequate to test special-token/padding invariants.
    class SyntheticTokenizer:
        def __call__(self, sequences, **kwargs):
            require(kwargs['truncation'] is False,'Synthetic truncation unexpectedly enabled')
            require(kwargs['return_special_tokens_mask'] is True,'Special-token mask not requested')
            width=max(len(x) for x in sequences)+2
            tok=torch.zeros((len(sequences),width),dtype=torch.long)
            att=torch.zeros_like(tok)
            special=torch.ones_like(tok)
            for i,seq in enumerate(sequences):
                n=len(seq)
                tok[i,:n+2]=torch.tensor([1]+[4]*n+[2])
                att[i,:n+2]=1
                special[i,1:n+1]=0
            return {'input_ids':tok,'attention_mask':att,'special_tokens_mask':special}
    ids,attention,mask,lengths=tokenize_residues(SyntheticTokenizer(),['AC','ACDE'],torch.device('cpu'))
    require(lengths.tolist()==[2,4] and mask.sum().item()==6,
            'Synthetic residue token masking failed')
    h=torch.arange(ids.numel()*5).reshape(ids.shape[0],ids.shape[1],5).float()
    h[0,0,:]=100000.0   # BOS must be excluded
    h[0,3,:]=-100000.0  # EOS must be excluded
    pooled=residue_features(h,mask,'M0')
    require(torch.allclose(pooled[0],h[0,1:3].mean(0)),'M0 FP32 masked-pool mismatch')
    pad=residue_features(h,mask,'M1')
    require(pad.shape==(2,4,5) and torch.equal(pad[0,2:],torch.zeros_like(pad[0,2:])),
            'M1/M3 feature padding mismatch')
    bad=type('TamperedTokenizer',(SyntheticTokenizer,),{
        '__call__':lambda self,*a,**kw: dict(input_ids=torch.tensor([[1,2,3]]),
        attention_mask=torch.tensor([[1,1,1]]),special_tokens_mask=torch.tensor([[0,0,0]]))
    })()
    try:
        tokenize_residues(bad,['AA'],torch.device('cpu'))
    except RuntimeError as e:
        require('Residue tokenization mismatch' in str(e),'Unexpected invalid-mask error')
    else:
        raise AssertionError('Tokenizer failure was not detected')
    print('SYNTHETIC TOKEN MASK / FP32 POOL / RAGGED FEATURE CHECKS: PASS')
    # Real original head classes, using supplied file hashes copied into package.
    for name in ('M0','M1','M3'):
        cfg=json_load(root/HOME[name]/'config.json')
        head=make_head(root,name,cfg)
        head.eval()
        with torch.inference_mode():
            if name=='M0':
                scores=head(torch.randn(2,1280))
            else:
                scores=head(torch.randn(2,5,1280),torch.tensor([3,5]))
        require(scores.shape==(2,) and torch.isfinite(scores).all(),
                f'Synthetic {name} forward invalid')
        print(f'{name}: ORIGINAL HEAD FORWARD / PARAM COUNT PASS')
    print('SYNTHETIC SELF-TEST PASS | No research project opened.')


def load_checkpoint_and_models(root, inv, proto, model, regime, device):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    candidate=next(c for c in inv['candidates'] if c['model']==model and c['regime']==regime)
    cfg=json_load(root/HOME[model]/'config.json')
    head=make_head(root,model,cfg).to(device)
    saved=torch.load(root/candidate['checkpoint_relative_path'],map_location='cpu',weights_only=True)
    require(saved['epoch']==candidate['selected_epoch'] and saved['backbone_commit']==ESM_REV,
            'Checkpoint metadata does not match selection')
    require(saved.get('selection_criterion','validation_loss')=='validation_loss',
            'Selected checkpoint criterion mismatch')
    if regime=='frozen':
        require(saved['config_sha256']==__import__('hashlib').sha256(
                json.dumps(cfg,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                'Frozen checkpoint config mismatch')
        head.load_state_dict(saved['model'],strict=True)
    else:
        require(saved['model_id']==model and saved['seed']==candidate['seed'] and
                saved['protocol_sha256']==FT_PROTOCOL_SHA and
                saved['baseline_sha256']==proto['models'][model]['checkpoint_sha256'] and
                saved['selection_criterion']=='validation_loss',
                'LoRA checkpoint metadata mismatch')
        head.load_state_dict(saved['head'],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(proto['backbone']['repository'],
                                            revision=ESM_REV,use_fast=True)
    encoder=AutoModel.from_pretrained(proto['backbone']['repository'],
                                      revision=ESM_REV,dtype=torch.bfloat16).to(device)
    encoder.requires_grad_(False)
    if regime=='LoRA':
        lc=LoraConfig(r=proto['lora']['rank'],lora_alpha=proto['lora']['alpha'],
                      lora_dropout=proto['lora']['dropout'],bias='none',
                      target_modules=proto['lora']['target_modules'])
        encoder=get_peft_model(encoder,lc)
        require(len(encoder.targeted_module_names)==66,
                'Expected 66 query/value target modules')
        result=set_peft_model_state_dict(encoder,saved['adapter'],adapter_name='default')
        require(len(result.unexpected_keys)==0,
                'PEFT adapter loader returned unexpected keys: '+str(result.unexpected_keys[:8]))
        loaded=get_peft_model_state_dict(encoder)
        require(set(loaded)==set(saved['adapter']),'PEFT adapter tensor key mismatch')
        for key,value in saved['adapter'].items():
            require(loaded[key].shape==value.shape and
                    torch.equal(loaded[key].cpu(),value.cpu()),
                    f'PEFT adapter tensor failed roundtrip: {key}')
        print('LoRA adapter re-extraction / tensor equality: PASS',flush=True)
    del saved
    encoder.eval();head.eval()
    require(not encoder.training and not head.training,'Evaluation mode incorrect')
    return tokenizer,encoder,head


def runtime_guard(proto):
    import torch
    from importlib.metadata import version
    require(torch.cuda.is_available(),'Real validation replay requires CUDA/L4; stay on CPU for preflight')
    require(torch.cuda.is_bf16_supported(),'CUDA BF16 support required')
    for name,expected in proto['software'].items():
        observed=torch.__version__ if name=='torch' else version(name)
        require(observed==expected,f'Pinned {name} mismatch: {observed} != {expected}')
    print(f'GPU: {torch.cuda.get_device_name(0)} | BF16 support PASS',flush=True)
    print('PINNED DEPENDENCY VERSIONS: PASS',flush=True)


def expected_record(root,model,regime,seed,epoch):
    import pandas as pd
    if regime=='frozen':
        path=root/'benchmark_v2_validation_addendum/summary/all_models_validation_metrics.csv'
        require(path.is_file(),'Frozen model validation summary missing')
        df=pd.read_csv(path)
        require({'Model','MCC','F1','AUROC','AP','Brier'}.issubset(df.columns),
                'Unexpected frozen validation summary columns')
        r=df.loc[df['Model']==model]
        require(len(r)==1,'Missing unique frozen validation model')
        row=r.iloc[0]
        return {k:float(row[column]) for k,column in [('mcc','MCC'),('f1','F1'),
             ('roc_auc','AUROC'),('average_precision','AP'),('brier','Brier')]}
    path=root/f'finetuning_v2/{model}_lora/seed{seed}/history.csv'
    df=pd.read_csv(path)
    r=df.loc[df['epoch']==epoch]
    require(len(r)==1,'Selected fine-tuning epoch not found')
    row=r.iloc[0]
    return {k:float(row[column]) for k,column in [('mcc','validation_mcc_0_5'),
          ('f1','validation_f1_0_5'),('roc_auc','validation_roc_auc'),
          ('average_precision','validation_average_precision'),('brier','validation_brier')]}


def calculate_metrics(y,p):
    import numpy as np
    from sklearn.metrics import (matthews_corrcoef, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix)
    y=np.asarray(y,dtype=np.int64)
    p=np.asarray(p,dtype=np.float64)
    require(len(y)==len(p)==1500 and np.isfinite(p).all() and
            np.all((p>=0)&(p<=1)),'Invalid validation rows/probabilities')
    yh=(p>=PREDICTION_THRESHOLD).astype(np.int64)
    cm=confusion_matrix(y,yh,labels=[0,1])
    tn,fp,fn,tp=map(int,cm.ravel())
    return {'mcc':float(matthews_corrcoef(y,yh)),
            'f1':float(f1_score(y,yh,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'brier':float(np.mean((p-y)**2)),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp}


def real_validation(root, model, regime):
    import numpy as np
    import torch
    import pandas as pd
    inv,proto=verified_preflight(root)
    verify_checkpoint_structures(root,inv)
    runtime_guard(proto)
    sys.path.insert(0,str(root/'training_v2'))
    from common import load_development
    parts=load_development(root) # Explicitly train + validation ONLY.
    val=parts['validation']
    require(len(val)==1500 and val['protein_id'].is_unique and
            (val['label'].value_counts().to_dict()=={0:750,1:750}) and
            (val['split'].astype(str)=='validation').all(),
            'Validation split integrity check failed')
    candidate=next(c for c in inv['candidates'] if c['model']==model and c['regime']==regime)
    device=torch.device('cuda:0')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    tokenizer,encoder,head=load_checkpoint_and_models(root,inv,proto,model,regime,device)
    probabilities=np.full(1500,np.nan,dtype=np.float64)
    seen=np.zeros(1500,dtype=np.int8)
    # Original LoRA validation ordering: full stable length sort, then 2-row batches.
    # For M0 sorting uses residue length + 2 ESM specials, same order.
    # Every 2-row batch is within the 2048-token M0 limit at max length 1014.
    order=np.argsort(val['length'].to_numpy(dtype=np.int64),kind='stable')
    require(len(order)==1500 and len(set(map(int,order)))==1500,
            'Validation index order invalid')
    with torch.inference_mode():
        for start in range(0,1500,2):
            idx=order[start:start+2]
            ids,attention,mask,lengths=tokenize_residues(tokenizer,
                      val.iloc[idx]['sequence'].astype(str).tolist(),device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=encoder(input_ids=ids,attention_mask=attention).last_hidden_state
                x=residue_features(hidden,mask,model)
                logits=head(x) if model=='M0' else head(x,lengths)
            require(logits.shape==(len(idx),) and torch.isfinite(logits).all(),
                    'Unexpected logits shape or nonfinite logits')
            pp=torch.sigmoid(logits.float()).detach().cpu().numpy().astype(np.float64)
            require(np.all(seen[idx]==0),'Duplicate prediction indices')
            probabilities[idx]=pp
            seen[idx]=1
            if (start+len(idx))%250==0 or start+len(idx)==1500:
                print(f'{model} {regime} validation: {start+len(idx)}/1500',flush=True)
    require(np.all(seen==1),'Incomplete validation replay')
    y=val['label'].to_numpy(dtype=np.int64)
    observed=calculate_metrics(y,probabilities)
    saved=expected_record(root,model,regime,candidate['seed'],candidate['selected_epoch'])
    deltas={metric:observed[metric]-saved[metric] for metric in saved}
    print('\nVALIDATION REPLAY COMPARISON',flush=True)
    print('Model:',model,'Regime:',regime,'Selected epoch:',candidate['selected_epoch'])
    for metric in ['mcc','f1','roc_auc','average_precision','brier']:
        print(f'{metric:>20}: recorded={saved[metric]:.9f} observed={observed[metric]:.9f} delta={deltas[metric]:+.9f}',flush=True)
    print('TN FP FN TP:',observed['tn'],observed['fp'],observed['fn'],observed['tp'])
    # Frozen baseline cache used FP16 storage; BF16 live encoder can change logits.
    # Never conceal a disagreement or automatically override the replay criteria.
    ok=(abs(deltas['mcc'])<=MCC_TOL and abs(deltas['f1'])<=MCC_TOL and
        abs(deltas['roc_auc'])<=AUC_AP_TOL and
        abs(deltas['average_precision'])<=AUC_AP_TOL and
        abs(deltas['brier'])<=BRIER_TOL)
    print('STRICT METRIC REPLAY:', 'PASS' if ok else 'REQUIRES INVESTIGATION',flush=True)
    # Local exploratory output only, NOT evaluation lock or heldout authorization.
    out=Path('validation_replay_output')
    out.mkdir(parents=True,exist_ok=True)
    suffix=f'{model}_{regime}_validation'
    dest=out/f'{suffix}.csv'
    meta=out/f'{suffix}.json'
    require(not dest.exists() and not meta.exists(),
            'Refusing to overwrite existing validation replay outputs')
    frame=pd.DataFrame({'protein_id':val['protein_id'].astype(str),
                        'label':y,'probability_amr':probabilities,
                        'predicted_at_0_5':(probabilities>=0.5).astype(np.int8)})
    frame.to_csv(dest,index=False,float_format='%.10g')
    record={'scope':'validation_only','model':model,'regime':regime,
            'checkpoint_sha256':candidate['checkpoint_sha256'],
            'checkpoint_epoch':candidate['selected_epoch'],
            'observed_metrics':observed,'reference_metrics':saved,'deltas':deltas,
            'strict_replay_pass':bool(ok),'predictions_csv_sha256':sha256(dest),
            'frozen_cache_live_precision_limitation':True,
            'source_sha256':sha256(Path(__file__)),'test_ood_opened':False}
    meta.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print('Local validation output:',dest)
    print('Local validation record:',meta)
    print('Test/OOD: NOT OPENED | Project files: UNMODIFIED',flush=True)
    require(ok,'Strict validation replay did not match the saved metrics; inspect before proceeding')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,
        default=Path('.'))
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test',action='store_true')
    modes.add_argument('--preflight',action='store_true')
    modes.add_argument('--validate',action='store_true')
    parser.add_argument('--model',choices=['M0','M1','M3'])
    parser.add_argument('--regime',choices=['frozen','LoRA'])
    args=parser.parse_args()
    require((args.model is not None and args.regime is not None) if args.validate else
            (args.model is None and args.regime is None),
            '--model and --regime are required only for --validate')
    if args.self_test:
        self_test()
    elif args.preflight:
        inv,_=verified_preflight(args.root)
        verify_checkpoint_structures(args.root,inv)
        print('READ-ONLY CPU PREFLIGHT: PASS')
    else:
        real_validation(args.root,args.model,args.regime)

if __name__=='__main__':
    main()
