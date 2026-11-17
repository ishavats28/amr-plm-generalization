#!/usr/bin/env python3
"""Train the M0 ESM-2 multilayer perceptron with LoRA using training and validation data only. Held-out partitions remain excluded.

Usage: python -u train.py --root <study-root> --seed 42
       python -u train.py --root ... --seed 42 --smoke
       python -u train.py --root ... --seed 42 --preflight
       python -u train.py --root ... --seed 42 --gpu-smoke
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Read-only modes must not generate Python bytecode in locked source directories.
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score)

AUDIT_SHA = 'f587a237fdf4a48b1beaa559cdd56e2461cf46b600b1038e3c5969eb29a6e8ac'
PROTOCOL_SHA = '7ee4e5f963403568e0adceebac98afdb029709b92b80ff3479177309a6459cad'
ESM_COMMIT = '08e4846e537177426273712802403f7ba8261b6c'
BASELINE_SHA = '52eedde296bc4059571a2b12dff34424e030983be422ae4555b93cc5e37aeda7'
M0_TRAIN_SHA = '916e0681c7c23490d5bd2db60fc9a1d62c64adf0cd6b070ccdb5ac4ff448bd19'
PARAMS_LORA = 1351680
PARAMS_HEAD = 344449
FIELDS = ['epoch','train_loss','validation_loss','validation_mcc_0_5',
          'validation_f1_0_5','validation_roc_auc','validation_average_precision',
          'validation_brier','validation_accuracy_0_5','validation_balanced_accuracy_0_5',
          'validation_precision_0_5','validation_recall_0_5','optimizer_steps',
          'epoch_seconds','best_epoch_so_far','epochs_without_improvement']


def require(ok, why):
    if not ok:
        raise RuntimeError('STOP: '+why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def digest_obj(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
          separators=(',',':')).encode('utf-8')).hexdigest()


def save_atomic(path, obj):
    partial=path.with_name(path.name+'.partial')
    require(not partial.exists(),f'stale partial: {partial}')
    with partial.open('xb') as f:
        torch.save(obj,f)
        f.flush(); os.fsync(f.fileno())
    os.replace(partial,path)


def write_atomic(path, contents):
    partial=path.with_name(path.name+'.partial')
    require(not partial.exists(),f'stale partial: {partial}')
    with partial.open('xb') as f:
        f.write(contents); f.flush(); os.fsync(f.fileno())
    os.replace(partial,path)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def check_peft_environment():
    """Reject known incompatible optional torchao before costly ESM-2 downloads."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version('torchao')
    except PackageNotFoundError:
        installed = None
    if installed is not None:
        from packaging.version import Version
        require(Version(installed) > Version('0.16.0'),
                f'incompatible optional torchao=={installed}; PEFT requires >0.16.0. '
                'Remove incompatible torchao from the runtime, then retry; '
                'do not change frozen model files.')


def load_inputs(root):
    base=root/'training_v2'
    sys.path.insert(0,str(base))
    from common import load_development
    require(sha(root/'metadata/v2_split_audit.json')==AUDIT_SHA,
            'certified split audit checksum changed')
    proto_path=root/'finetuning_v2/protocol_v1.json'
    require(proto_path.is_file() and sha(proto_path)==PROTOCOL_SHA,
            'locked fine-tuning protocol changed')
    proto=json.loads(proto_path.read_text())
    require(proto['status']=='locked' and proto['protocol_id']=='AMR_V2_ESM2_LORA_FT_001' and
            proto['audit_sha256']==AUDIT_SHA and proto['backbone']['commit']==ESM_COMMIT and
            proto['dataset_version']=='V2' and proto['execution_order']==['M1','M3','M0'] and
            proto['test_ood_access']=='prohibited_until_final_evaluation',
            'protocol identity/backbone/data boundaries unexpected')
    tr=proto['training']; lr=proto['lora']
    require(tr['adapter_lr']==0.0001 and tr['head_lr']==0.00001 and
            tr['microbatch_size']==2 and tr['gradient_accumulation_steps']==8 and
            tr['effective_batch_size']==16 and tr['max_epochs']==8 and
            tr['early_stopping_patience']==3 and tr['early_stopping_min_delta']==0.0001 and
            tr['checkpoint_selection']=='validation_loss' and tr['seeds']==[42,52,62] and
            tr['no_sequence_truncation'] and tr['scheduler']=='cosine_decay' and
            tr['warmup_fraction']==0.05 and tr['weight_decay']==0.01 and
            tr['gradient_clip_norm']==1.0 and tr['optimizer']=='AdamW' and
            tr['initialization']=='selected_frozen_baseline_head_plus_zero_initialized_lora' and
            tr['classification_threshold']==0.5 and
            lr['rank']==8 and lr['alpha']==16 and lr['dropout']==0.05 and
            lr['bias']=='none' and lr['expected_target_count']==66 and
            lr['trainable_adapter_parameters']==PARAMS_LORA and
            lr['target_modules']==['query','value'],
            'protocol training/LoRA settings unexpected')
    freeze=root/'benchmark_v2_freeze/PACKAGE_MANIFEST.json'
    val=root/'benchmark_v2_validation_addendum/summary/FINAL_MANIFEST.json'
    require(sha(freeze)==proto['baseline_freeze_sha256'] and
            sha(val)==proto['validation_addendum_sha256'],
            'frozen benchmark or addendum changed')
    # M1 and M0 must already be completed and unchanged before the final M0 architecture.
    for model_id in ('M1','M3'):
        prev_root=root/f'finetuning_v2/{model_id}_lora'
        expected_baseline=proto['models'][model_id]['checkpoint_sha256']
        for previous_seed in tr['seeds']:
            run_prev=prev_root/f'seed{previous_seed}'
            summary_path=run_prev/'training_summary.json'
            require(summary_path.is_file(),
                    f'{model_id} seed {previous_seed} completion audit missing')
            rec=json.loads(summary_path.read_text())
            require(rec.get('status')=='complete' and rec.get('model_id')==model_id and
                    rec.get('seed')==previous_seed and rec.get('dataset_version')=='V2' and
                    rec.get('protocol_sha256')==PROTOCOL_SHA and
                    rec.get('baseline_checkpoint_sha256')==expected_baseline and
                    rec.get('test_ood_accessed') is False and
                    rec.get('selection_criterion')=='validation_loss',
                    f'{model_id} seed {previous_seed} completion provenance changed')
            for name,key in [('best_model.pt','best_model_sha256'),
                             ('last_checkpoint.pt','last_checkpoint_sha256'),
                             ('history.csv','history_sha256')]:
                require(sha(run_prev/name)==rec[key],
                        f'{model_id} seed {previous_seed} output hash mismatch: {name}')
    from importlib.metadata import version
    for package, expected_version in proto['software'].items():
        observed = torch.__version__ if package == 'torch' else version(package)
        require(observed == expected_version,
                f'locked {package} version mismatch: {observed} vs {expected_version}')
    report=json.loads(val.read_text())
    require(report['status']=='complete' and report['validation_only'] is True and
            report['test_ood_read'] is False and
            report['model_ids']==[f'M{i}' for i in range(8)],
            'benchmark validation status unexpected')
    m0=base/'M0_esm2_mlp'
    record=proto['models']['M0']
    require(sha(m0/'train.py')==M0_TRAIN_SHA and
            record['source_sha256']==M0_TRAIN_SHA,
            'baseline M0 classifier implementation changed')
    ckpt=m0/'runs/seed42/best_model.pt'
    require(sha(ckpt)==BASELINE_SHA and record['checkpoint_sha256']==BASELINE_SHA,
            'frozen M0 selected checkpoint mismatch')
    cfg_path=m0/'config.json'
    cfg=json.loads(cfg_path.read_text())
    require(cfg['model_id']=='M0' and cfg['architecture']=='frozen_esm2_residue_mean_mlp' and
            cfg['backbone']=='facebook/esm2_t33_650M_UR50D' and
            cfg['backbone_revision']==ESM_COMMIT and cfg['hidden_size']==1280 and
            cfg['head_widths']==[256,64] and cfg['dropout']==0.3 and
            cfg['pooling']=='mean over residues excluding special and padding tokens; FP32 accumulation' and
            cfg['embedding_max_padded_tokens_per_batch']==2048,
            'baseline M0 encoder/head/pooling configuration changed')
    require(sha(cfg_path)==record['config_sha256'] and
            record['head_trainable_parameters']==PARAMS_HEAD,
            'baseline M0 source or config differs from locked protocol')
    frames=load_development(root)  # ONLY certified train and validation files
    require(len(frames['train'])==proto['train_rows']==12152 and
            len(frames['validation'])==proto['validation_rows']==1500,
            'certified training and validation counts differ')
    return proto,cfg,ckpt,m0,frames


def load_head(m0,cfg,ckpt,device='cuda'):
    # Import the verified ORIGINAL M0Head; no reconstructed or altered architecture.
    spec=importlib.util.spec_from_file_location('amr_m0_ft_baseline_head',m0/'train.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    head=mod.M0Head(tuple(cfg['head_widths']),float(cfg['dropout'])).to(device)
    require(sum(p.numel() for p in head.parameters())==PARAMS_HEAD,
            'M0 head trainable parameter count mismatch')
    saved=torch.load(ckpt,map_location='cpu',weights_only=True)
    require(saved['epoch']==6 and saved['backbone_commit']==ESM_COMMIT and
            saved['config_sha256']==digest_obj(cfg) and
            'model' in saved,
            'M0 baseline checkpoint provenance unexpected')
    head.load_state_dict(saved['model'],strict=True)
    return head


def get_tokens(tokenizer, df, ids, device='cuda'):
    data=[str(df.iloc[int(i)]['sequence']) for i in ids]
    require(all(1<=len(s)<=1014 for s in data),'out-of-range protein length')
    p=tokenizer(data,return_tensors='pt',padding=True,truncation=False,
                return_special_tokens_mask=True)
    tokens=p['input_ids'].to(device)
    attention=p['attention_mask'].to(device)
    keep=attention.bool() & ~p['special_tokens_mask'].to(device).bool()
    lengths=torch.as_tensor([len(s) for s in data],device=device,dtype=torch.long)
    require(tokens.shape[1]<=1024,'ESM2 token limit exceeded')
    require(torch.equal(keep.sum(dim=1),lengths),'tokenization-residue mismatch')
    labels=torch.as_tensor([float(df.iloc[int(i)]['label']) for i in ids],device=device,dtype=torch.float32)
    return tokens,attention,keep,lengths,labels


def features(hidden,mask):
    # Original M0 extraction semantics: strictly residue-only, FP32 accumulation.
    keep=mask.unsqueeze(-1)
    numerator=(hidden.float()*keep).sum(dim=1)
    denominator=keep.sum(dim=1)
    require(bool((denominator>0).all()),'empty residue mask in M0 pooled features')
    return numerator/denominator


def metrics(y,p):
    y=np.asarray(y,dtype=np.int64)
    p=np.asarray(p,dtype=np.float64)
    yh=(p>=.5).astype(np.int64)
    return dict(
      mcc=float(matthews_corrcoef(y,yh)),
      f1=float(f1_score(y,yh,zero_division=0)),
      roc_auc=float(roc_auc_score(y,p)),
      average_precision=float(average_precision_score(y,p)),
      brier=float(np.mean((p-y)**2)),
      accuracy=float(accuracy_score(y,yh)),
      balanced_accuracy=float(balanced_accuracy_score(y,yh)),
      precision=float(precision_score(y,yh,zero_division=0)),
      recall=float(recall_score(y,yh,zero_division=0)))


def schedule_lambda(step,warmup,total):
    require(total>0 and warmup>=0 and step>=0,'invalid schedule counters')
    if warmup and step<warmup:
        return float(step+1)/float(warmup)
    pos=min(1.0,max(0.0, (step-warmup)/max(1,total-warmup)))
    return 0.5*(1+math.cos(math.pi*pos))


def group_batches(batches,n):
    for start in range(0,len(batches),n):
        yield batches[start:start+n]


def make_batches(indices,lengths,max_items,budget,*,shuffle,rng):
    # Same length-bucket strategy as the verified frozen M0 trainer, with a
    # M0-specific upper bound in padded ESM input TOKENS (residues + BOS/EOS).
    indices=np.asarray(indices,dtype=np.int64)
    if shuffle:
        indices=indices[torch.randperm(len(indices),generator=rng).numpy()]
    result=[]
    block_size=256 if shuffle else len(indices)
    for start in range(0,len(indices),block_size):
        block=indices[start:start+block_size]
        block=block[np.argsort(lengths[block],kind='stable')]
        cursor=0
        while cursor<len(block):
            end,max_len=cursor,0
            while end<len(block) and end-cursor<max_items:
                proposed=max(max_len,int(lengths[block[end]]))
                if end>cursor and proposed*(end-cursor+1)>budget:break
                end+=1;max_len=proposed
            require(end>cursor,'empty M0 batch')
            require(max_len*(end-cursor)<=budget,'M0 padded input token budget exceeded')
            result.append(block[cursor:end])
            cursor=end
    if shuffle:
        order=torch.randperm(len(result),generator=rng).tolist()
        result=[result[j] for j in order]
    require(sum(len(part) for part in result)==len(indices),'M0 batch coverage failure')
    require(len(set(int(i) for b in result for i in b))==len(indices),
            'M0 duplicate training/validation row in batches')
    return result


def row_history(history):
    import io
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=FIELDS)
    writer.writeheader();writer.writerows(history)
    return buffer.getvalue().encode()


def ensure_no_partials(run):
    require(not any(p.name.endswith(('.partial','.tmp')) for p in run.glob('*')),
            'orphaned partial output in fine-tuning run')


def smoke_local():
    # CPU-only, fully synthetic functional tests; do not read train/validation.
    assert abs(schedule_lambda(0,5,100)-.2)<1e-12
    assert abs(schedule_lambda(4,5,100)-1)<1e-12
    assert abs(schedule_lambda(100,5,100))<1e-12
    assert [list(map(len,x)) for x in group_batches([[0],[1,2],[3]],2)]==[[1,2],[1]]
    probs=np.array([.9,.1,.8,.2]);labels=np.array([1,0,1,0])
    m=metrics(labels,probs)
    require(abs(m['mcc']-1)<1e-12 and abs(m['accuracy']-1)<1e-12,
            'synthetic metrics incorrect')
    # Verify gradient accumulation sample weighting for uneven final group.
    a=torch.nn.Parameter(torch.tensor(0.1))
    x=[torch.tensor([1.,2.]),torch.tensor([3.])]
    y=[torch.tensor([1.,0.]),torch.tensor([1.])]
    for xa,ya in zip(x,y):
        (nn.functional.binary_cross_entropy_with_logits(a*xa,ya,reduction='sum')/3).backward()
    ref=torch.nn.Parameter(torch.tensor(0.1))
    nn.functional.binary_cross_entropy_with_logits(ref*torch.cat(x),torch.cat(y)).backward()
    require(torch.allclose(a.grad,ref.grad,atol=1e-7),
            'gradient accumulation sample weighting incorrect')
    # Test length batching and full coverage with the actual locked batch size.
    rng=torch.Generator(device='cpu').manual_seed(42)
    arr=np.array([75,81,100,1014,235,800,1012,512],dtype=np.int64)+2
    batches=make_batches(np.arange(8),arr,2,2048,shuffle=True,rng=rng)
    require(len(batches)==4 and all(len(z)==2 for z in batches),
            'M0 synthetic microbatch construction incorrect')
    require(all(int(arr[z].max())*len(z)<=2048 for z in batches),
            'M0 synthetic padded-token budget exceeded')
    # Import original verified M0 head; verify exact topology and gradients.
    root_m0=Path(__file__).resolve().parents[2]/'training_v2'/'M0_esm2_mlp'
    require(root_m0.is_dir() and sha(root_m0/'train.py')==M0_TRAIN_SHA,
            'original M0 source missing or changed for synthetic smoke')
    sys.path.insert(0,str(root_m0.parent))
    spec=importlib.util.spec_from_file_location('amr_m0_lora_smoke',root_m0/'train.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    cfg=json.loads((root_m0/'config.json').read_text())
    torch.manual_seed(78)
    head=base.M0Head(tuple(cfg['head_widths']),float(cfg['dropout'])).cpu()
    require(sum(p.numel() for p in head.parameters())==PARAMS_HEAD,
            'M0 synthetic head parameter count mismatch')
    h=torch.randn(2,19,1280,requires_grad=True)
    keep=torch.zeros(2,19,dtype=torch.bool)
    keep[0,:11]=True;keep[1,:19]=True
    head.eval()
    with torch.no_grad():
        initial=head(features(h,keep))
        extended=torch.cat((h,torch.full((2,7,1280),50.0)),dim=1)
        keep_extended=torch.cat((keep,torch.zeros((2,7),dtype=torch.bool)),dim=1)
        padded=head(features(extended,keep_extended))
        require(torch.allclose(initial,padded,atol=1e-6,rtol=1e-6),
                'M0 masked mean pooling/padding invariance failed')
    head.train();head.zero_grad(set_to_none=True)
    loss=nn.functional.binary_cross_entropy_with_logits(
        head(features(h,keep)),torch.tensor([0.,1.]))
    loss.backward()
    require(h.grad is not None and torch.isfinite(h.grad).all() and
            bool((h.grad[0,:11].abs().max()>0).item()) and
            bool((h.grad[0,11:]==0).all().item()),
            'M0 synthetic pooling gradient/masking incorrect')
    require(all(p.grad is not None and torch.isfinite(p.grad).all()
                for p in head.parameters()),'M0 synthetic head gradients missing')
    print('M0 ORIGINAL MLP HEAD + FP32 RESIDUE MEAN POOL SMOKE: PASS',flush=True)
    print('Optimizer schedule, sample-weighted accumulation, batching, metrics: PASS')
    print('No project files written. Test/OOD not read.',flush=True)


def preflight(root,seed):
    require(seed in (42,52,62),'seed must be from locked protocol')
    check_peft_environment()
    proto,cfg,ckpt,m0,frames=load_inputs(root)
    _head=load_head(m0,cfg,ckpt,device='cpu')
    run=root/f'finetuning_v2/M0_lora/seed{seed}'
    require(not (run/'training_summary.json').exists(),
            'M0 LoRA seed run already complete: cannot overwrite')
    if run.exists():
        ensure_no_partials(run)
        require(not any((run/f).exists() for f in ('best_model.pt','history.csv')) or
                (run/'last_checkpoint.pt').is_file(),
                'orphaned M0 run output: inspect before proceeding')
    print('M0 LORA READ-ONLY PREFLIGHT: PASS',flush=True)
    print(f'Seed: {seed} | train={len(frames["train"])} validation={len(frames["validation"])}',flush=True)
    print('Verified original M0 head:',sum(p.numel() for p in _head.parameters()),flush=True)
    print('M1/M3 completed seeds, benchmark hashes, locked protocol: PASS',flush=True)
    print('Research files unmodified. Test/OOD not accessed.',flush=True)


def main(root,seed,smoke,gpu_smoke=False,do_preflight=False):
    if smoke:
        smoke_local(); return
    if do_preflight:
        preflight(root,seed); return
    from transformers import AutoTokenizer,AutoModel
    from peft import LoraConfig,get_peft_model
    from peft.utils import get_peft_model_state_dict,set_peft_model_state_dict
    require(seed in (42,52,62),'seed must be from locked protocol')
    check_peft_environment()
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'GPU BF16 required')
    require(torch.cuda.get_device_properties(0).total_memory>=18*2**30,
            'GPU requires >=18 GiB')
    proto,cfg,ckpt,m0,frames=load_inputs(root)
    params=proto['training']
    run=root/f'finetuning_v2/M0_lora/seed{seed}'
    require(not (run/'training_summary.json').exists(),
            'M0 LoRA seed run already complete: cannot overwrite')
    if run.exists():ensure_no_partials(run)
    require(not any((run/f).exists() for f in ('best_model.pt','history.csv')) or
            (run/'last_checkpoint.pt').exists(),
            'orphaned run output: inspect before proceeding')
    trainer_sha=sha(Path(__file__).resolve())
    train,val=frames['train'],frames['validation']
    seed_everything(seed)
    device='cuda'
    print(f'M0 LORA seed={seed} | training {len(train)} validation {len(val)}',flush=True)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg['backbone'],revision=ESM_COMMIT,use_fast=True)
    encoder=AutoModel.from_pretrained(cfg['backbone'],revision=ESM_COMMIT,dtype=torch.bfloat16).to(device)
    encoder.requires_grad_(False);encoder.eval()
    head=load_head(m0,cfg,ckpt)
    head.eval()
    # Before any optimizer updates: same live tensors, before vs after zero LoRA adapter.
    ids=[int(train['length'].idxmin()),int(train['length'].idxmax())]
    tokens,attention,keep,lengths,targets=get_tokens(tokenizer,train,ids)
    with torch.inference_mode():
        frozen_logits=head(features(encoder(input_ids=tokens,attention_mask=attention).last_hidden_state,keep)).detach().clone()
    lora=LoraConfig(r=proto['lora']['rank'],lora_alpha=proto['lora']['alpha'],
                    lora_dropout=proto['lora']['dropout'],bias='none',
                    target_modules=proto['lora']['target_modules'])
    encoder=get_peft_model(encoder,lora)
    require(len(encoder.targeted_module_names)==66,
            'ESM2 LoRA target count mismatch')
    adapt=[p for p in encoder.parameters() if p.requires_grad]
    hp=[p for p in head.parameters() if p.requires_grad]
    require(sum(p.numel() for p in adapt)==PARAMS_LORA and
            sum(p.numel() for p in hp)==PARAMS_HEAD and
            all('lora_' in n for n,p in encoder.named_parameters() if p.requires_grad),
            'LoRA/head parameter-freeze mismatch')
    encoder.eval()
    with torch.inference_mode():
        after=head(features(encoder(input_ids=tokens,attention_mask=attention).last_hidden_state,keep)).detach()
    max_delta=float((after-frozen_logits).abs().max())
    require(max_delta<=0.02,'LoRA adapter initialization changed logits')
    print(f'M0 LoRA zero-adapter logit parity max_delta={max_delta:.8f} PASS',flush=True)
    del tokens,attention,keep,lengths,targets,after,frozen_logits
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(encoder,'enable_input_require_grads'):
        encoder.enable_input_require_grads()
    train_rng=torch.Generator(device='cpu').manual_seed(seed)
    optimizer=torch.optim.AdamW([
        {'params':adapt,'lr':params['adapter_lr']},
        {'params':hp,'lr':params['head_lr']}],
        weight_decay=params['weight_decay'])
    if gpu_smoke:
        # Real end-to-end GPU smoke, training data only, no project files written.
        import io
        encoder.train();head.train()
        ids=[int(train['length'].idxmin()),int(train['length'].idxmax())]
        tokens,attention,keep,lengths,targets=get_tokens(tokenizer,train,ids)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=encoder(input_ids=tokens,attention_mask=attention).last_hidden_state
            logits=head(features(hidden,keep))
            loss=nn.functional.binary_cross_entropy_with_logits(logits.float(),targets)
        require(bool(torch.isfinite(loss)),'M0 GPU smoke nonfinite loss')
        loss.backward()
        require(any(p.grad is not None and torch.isfinite(p.grad).all() and
                    bool((p.grad.abs().max()>0).item()) for p in adapt),
                'M0 GPU smoke no LoRA gradients')
        require(any(p.grad is not None and torch.isfinite(p.grad).all() and
                    bool((p.grad.abs().max()>0).item()) for p in hp),
                'M0 GPU smoke no head gradients')
        torch.nn.utils.clip_grad_norm_(adapt+hp,max_norm=1.0)
        optimizer.step()
        # Save/load exactly the adapter/head payload used by training, in memory.
        state={'adapter':get_peft_model_state_dict(encoder),'head':head.state_dict()}
        buff=io.BytesIO();torch.save(state,buff);buff.seek(0)
        loaded=torch.load(buff,map_location='cpu',weights_only=True)
        require(set(state['adapter'])==set(loaded['adapter']) and
                set(state['head'])==set(loaded['head']),
                'M0 GPU smoke adapter/head checkpoint roundtrip failed')
        print(f'M0 GPU TRAINING SMOKE: PASS | loss={loss.item():.5f} '
              f'| peak allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB '
              f'| peak reserved={torch.cuda.max_memory_reserved()/2**30:.2f} GiB',flush=True)
        print('LoRA gradients + head gradients + optimizer + in-memory checkpoint: PASS',flush=True)
        print('No project files written. Test/OOD not opened.',flush=True)
        return
    # Two proteins per microbatch, bucketed by length, no truncation.
    lens=train['length'].to_numpy(dtype=np.int64)+2  # ESM BOS + EOS
    vlens=val['length'].to_numpy(dtype=np.int64)+2
    budget=int(cfg['embedding_max_padded_tokens_per_batch'])
    require(budget==2048,'M0 ESM input token budget changed')
    def make_train_batches():
        return make_batches(np.arange(len(train)),lens,
            int(params['microbatch_size']),budget,shuffle=True,rng=train_rng)
    # Exactly 6076 two-sequence microbatches for certified training split.
    n_micro=int(math.ceil(len(train)/params['microbatch_size']))
    require(n_micro==6076,'unexpected microbatch count')
    updates=int(math.ceil(n_micro/params['gradient_accumulation_steps']))
    total_steps=updates*params['max_epochs']
    warmup=int(math.ceil(total_steps*params['warmup_fraction']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step:schedule_lambda(step,warmup,total_steps))
    val_batches=make_batches(np.arange(len(val)),vlens,
        int(params['microbatch_size']),budget,shuffle=False,rng=train_rng)
    require(len(val_batches)==750,'unexpected M0 validation batch count')
    require(sum(map(len,val_batches))==len(val),'validation batch coverage incomplete')
    history=[]; best_epoch=0; best_loss=float('inf'); patience=0
    update_count=0; epoch_start=1; start=time.monotonic()
    if (run/'last_checkpoint.pt').exists():
        last=torch.load(run/'last_checkpoint.pt',map_location='cpu',weights_only=True)
        require(last['protocol_sha256']==PROTOCOL_SHA and
                last['baseline_sha256']==BASELINE_SHA and
                last['seed']==seed and last['model_id']=='M0' and
                last['trainer_sha256']==trainer_sha and
                last['backbone_commit']==ESM_COMMIT,
                'resumption checkpoint provenance changed')
        set_peft_model_state_dict(encoder,last['adapter'],adapter_name='default')
        head.load_state_dict(last['head'],strict=True)
        optimizer.load_state_dict(last['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to('cuda')
        scheduler.load_state_dict(last['scheduler'])
        train_rng.set_state(last['train_generator_state'])
        torch.set_rng_state(last['torch_rng_state'])
        torch.cuda.set_rng_state_all(last['cuda_rng_state'])
        history=list(last['history']); best_epoch=last['best_epoch'];best_loss=last['best_loss']
        patience=last['patience_count'];update_count=last['optimizer_steps']
        epoch_start=int(last['epoch'])+1
        require(len(history)==epoch_start-1,'saved history/epoch mismatch')
        require((run/'best_model.pt').is_file(),'resumption best checkpoint missing')
        best_saved=torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
        require(best_saved['epoch']==best_epoch and best_saved['model_id']=='M0' and
                best_saved['seed']==seed and
                best_saved['protocol_sha256']==PROTOCOL_SHA and
                best_saved['baseline_sha256']==BASELINE_SHA and
                best_saved['trainer_sha256']==trainer_sha and
                abs(float(best_saved['validation_loss'])-float(best_loss))<1e-10,
                'best checkpoint metadata inconsistent with last checkpoint')
        print('Resuming after epoch',last['epoch'],flush=True)
    else:
        # Refuse to create run files until input/provenance and parity have passed.
        run.mkdir(parents=True,exist_ok=True)
        require(not any(run.iterdir()),'nonempty new seed run')
    require(int(params['effective_batch_size'])==int(params['microbatch_size'])*int(params['gradient_accumulation_steps']),
            'effective batch mismatch')
    print(f'LoRA trainable={PARAMS_LORA:,}; head={PARAMS_HEAD:,}; effective batch={params["effective_batch_size"]}',flush=True)
    print(f'Optimizer steps per epoch={updates}; total budget={total_steps}; warmup={warmup}',flush=True)
    print('Checkpoint: minimum validation loss | test/OOD not loaded',flush=True)

    def evaluate():
        encoder.eval();head.eval()
        loss_sum=0.0
        predictions=np.empty(len(val),dtype=np.float64)
        with torch.inference_mode():
            for indices in val_batches:
                t,a,k,ln,y=get_tokens(tokenizer,val,indices)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=encoder(input_ids=t,attention_mask=a).last_hidden_state
                    logits=head(features(hidden,k))
                loss_sum+=float(nn.functional.binary_cross_entropy_with_logits(logits.float(),y,reduction='sum').item())
                predictions[np.asarray(indices,dtype=np.int64)]=torch.sigmoid(logits.float()).cpu().numpy()
        actual=val['label'].to_numpy(dtype=np.int64)
        require(np.isfinite(predictions).all(),'nonfinite validation probabilities')
        return {'loss':loss_sum/len(val),**metrics(actual,predictions)}

    stop_reason='max_epochs'
    for epoch in range(epoch_start,int(params['max_epochs'])+1):
        if patience>=int(params['early_stopping_patience']):
            stop_reason='early_stopping'; break
        tic=time.monotonic()
        encoder.train();head.train()
        batches=make_train_batches()
        require(sum(map(len,batches))==len(train),'training batch coverage incomplete')
        total_loss=0.0
        for group in group_batches(batches,int(params['gradient_accumulation_steps'])):
            denominator=sum(len(b) for b in group)
            require(1<=denominator<=int(params['effective_batch_size']),
                    'invalid gradient accumulation group')
            optimizer.zero_grad(set_to_none=True)
            for ids in group:
                t,a,k,ln,y=get_tokens(tokenizer,train,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=encoder(input_ids=t,attention_mask=a).last_hidden_state
                    logits=head(features(hidden,k))
                    batch_sum=nn.functional.binary_cross_entropy_with_logits(logits.float(),y,reduction='sum')
                    loss=batch_sum/denominator
                require(bool(torch.isfinite(loss).item()),'nonfinite training loss')
                loss.backward()
                total_loss+=float(batch_sum.detach().item())
                del hidden,logits,t,a,k,ln,y,loss,batch_sum
            norm=torch.nn.utils.clip_grad_norm_(adapt+hp,max_norm=float(params['gradient_clip_norm']))
            require(bool(torch.isfinite(norm)),'nonfinite gradient norm')
            optimizer.step(); scheduler.step(); update_count+=1
        val_metric=evaluate()
        require(math.isfinite(val_metric['loss']),'nonfinite validation loss')
        improvement=val_metric['loss']<best_loss-float(params['early_stopping_min_delta'])
        if improvement:
            best_loss=float(val_metric['loss']);best_epoch=epoch;patience=0
            save_atomic(run/'best_model.pt',{
                'epoch':epoch,'model_id':'M0','seed':seed,
                'adapter':get_peft_model_state_dict(encoder),
                'head':head.state_dict(),
                'protocol_sha256':PROTOCOL_SHA,
                'audit_sha256':AUDIT_SHA,
                'baseline_sha256':BASELINE_SHA,
                'backbone_commit':ESM_COMMIT,
                'validation_loss':best_loss,'selection_criterion':'validation_loss',
                'trainer_sha256':trainer_sha})
        else:
            patience+=1
        record=dict(epoch=epoch,train_loss=total_loss/len(train),
            validation_loss=val_metric['loss'],validation_mcc_0_5=val_metric['mcc'],
            validation_f1_0_5=val_metric['f1'],validation_roc_auc=val_metric['roc_auc'],
            validation_average_precision=val_metric['average_precision'],
            validation_brier=val_metric['brier'],validation_accuracy_0_5=val_metric['accuracy'],
            validation_balanced_accuracy_0_5=val_metric['balanced_accuracy'],
            validation_precision_0_5=val_metric['precision'],
            validation_recall_0_5=val_metric['recall'],optimizer_steps=update_count,
            epoch_seconds=time.monotonic()-tic,best_epoch_so_far=best_epoch,
            epochs_without_improvement=patience)
        history.append(record)
        save_atomic(run/'last_checkpoint.pt',{
            'epoch':epoch,'model_id':'M0','seed':seed,
            'adapter':get_peft_model_state_dict(encoder),
            'head':head.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'protocol_sha256':PROTOCOL_SHA,
            'trainer_sha256':trainer_sha,
            'baseline_sha256':BASELINE_SHA,'backbone_commit':ESM_COMMIT,
            'history':history,'best_epoch':best_epoch,'best_loss':best_loss,
            'patience_count':patience,'optimizer_steps':update_count,
            'train_generator_state':train_rng.get_state(),
            'torch_rng_state':torch.get_rng_state(),
            'cuda_rng_state':torch.cuda.get_rng_state_all()})
        write_atomic(run/'history.csv',row_history(history))
        print(f'Epoch {epoch:02d}/{params["max_epochs"]}: train_loss={record["train_loss"]:.5f} '
              f'val_loss={val_metric["loss"]:.5f} val_MCC={val_metric["mcc"]:.4f} '
              f'best={best_epoch} unimproved={patience} steps={update_count}',flush=True)
    if patience>=int(params['early_stopping_patience']):stop_reason='early_stopping'
    require(best_epoch>=1,'missing best epoch')
    require((run/'best_model.pt').is_file() and (run/'last_checkpoint.pt').is_file() and
            len(history)>0,'missing model checkpoint or history')
    best_record=next(x for x in history if x['epoch']==best_epoch)
    require(abs(best_record['validation_loss']-best_loss)<1e-10,'best record mismatch')
    out={'status':'complete','dataset_version':'V2','model_id':'M0','adaptation':'LoRA',
         'seed':seed,'train_rows':len(train),'validation_rows':len(val),
         'baseline_checkpoint_sha256':BASELINE_SHA,
         'protocol_sha256':PROTOCOL_SHA,'audit_sha256':AUDIT_SHA,
         'trainer_sha256':trainer_sha,
         'backbone_commit':ESM_COMMIT,'best_epoch':best_epoch,
         'epochs_executed':len(history),'stop_reason':stop_reason,
         'selection_criterion':'validation_loss',
         'best_validation_loss':best_loss,'best_validation_metrics':best_record,
         'best_model_sha256':sha(run/'best_model.pt'),
         'last_checkpoint_sha256':sha(run/'last_checkpoint.pt'),
         'history_sha256':sha(run/'history.csv'),
         'test_ood_accessed':False,'elapsed_seconds':time.monotonic()-start}
    write_atomic(run/'training_summary.json',json.dumps(out,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(f'M0 LORA TRAINING COMPLETE | seed {seed} | best epoch {best_epoch}',flush=True)
    print('Summary:',run/'training_summary.json',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--gpu-smoke',action='store_true')
    ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args()
    require(sum([args.smoke,args.gpu_smoke,args.preflight])<=1,
            'choose only one of --smoke, --gpu-smoke, or --preflight')
    main(args.root,args.seed,args.smoke,args.gpu_smoke,args.preflight)
