#!/usr/bin/env python3
"""Train the M3 ESM-2 BiLSTM–Transformer with LoRA using training and validation data only. Held-out partitions remain excluded.

Usage: python -u train.py --root <study-root> --seed 42
       python -u train.py --root ... --seed 42 --smoke
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

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score)

AUDIT_SHA = 'f587a237fdf4a48b1beaa559cdd56e2461cf46b600b1038e3c5969eb29a6e8ac'
PROTOCOL_SHA = '7ee4e5f963403568e0adceebac98afdb029709b92b80ff3479177309a6459cad'
ESM_COMMIT = '08e4846e537177426273712802403f7ba8261b6c'
BASELINE_SHA = 'beec5ad0438c2b7f30925de812026b370e621b53e738dab4bea5aa0f7816bb73'
M3_MODEL_SHA = '52855ceca628162171d8b0a4e04cf536c923c8a896448f091be23ffee736c46d'
M3_TRAIN_SHA = '72cb2667f5d072f9885bb97f2c89ddf135221d02bc4600dd756ab2e20c7b6fb9'
PARAMS_LORA = 1351680
PARAMS_HEAD = 1927042
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


def load_inputs(root, smoke):
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
            proto['audit_sha256']==AUDIT_SHA and proto['backbone']['commit']==ESM_COMMIT,
            'protocol identity/backbone unexpected')
    tr=proto['training']; lr=proto['lora']
    require(tr['adapter_lr']==0.0001 and tr['head_lr']==0.00001 and
            tr['microbatch_size']==2 and tr['gradient_accumulation_steps']==8 and
            tr['max_epochs']==8 and tr['early_stopping_patience']==3 and
            tr['checkpoint_selection']=='validation_loss' and tr['seeds']==[42,52,62] and
            tr['no_sequence_truncation'] and
            lr['rank']==8 and lr['alpha']==16 and lr['dropout']==0.05 and
            lr['target_modules']==['query','value'],
            'protocol training/LoRA settings unexpected')
    freeze=root/'benchmark_v2_freeze/PACKAGE_MANIFEST.json'
    val=root/'benchmark_v2_validation_addendum/summary/FINAL_MANIFEST.json'
    require(sha(freeze)==proto['baseline_freeze_sha256'] and
            sha(val)==proto['validation_addendum_sha256'],
            'frozen benchmark or addendum changed')
    # Confirm previous M1 architecture finished all three predefined seeds.
    m1_root=root/'finetuning_v2/M1_lora'
    for m1_seed in (42,52,62):
        run_m1=m1_root/f'seed{m1_seed}'
        summary_path=run_m1/'training_summary.json'
        require(summary_path.is_file(),f'M1 seed {m1_seed} completion audit missing')
        record=json.loads(summary_path.read_text())
        require(record.get('status')=='complete' and record.get('model_id')=='M1' and
                record.get('seed')==m1_seed and record.get('dataset_version')=='V2' and
                record.get('protocol_sha256')==PROTOCOL_SHA and
                record.get('baseline_checkpoint_sha256')==
                  'aa59ef5cc03cfccfbc30f13fc780a119ab79f4e5271cb31bc59d341635cbb3e3' and
                record.get('test_ood_accessed') is False and
                record.get('selection_criterion')=='validation_loss',
                f'M1 seed {m1_seed} completion provenance changed')
        for name,key in [('best_model.pt','best_model_sha256'),
                         ('last_checkpoint.pt','last_checkpoint_sha256'),
                         ('history.csv','history_sha256')]:
            require(sha(run_m1/name)==record[key],
                    f'M1 seed {m1_seed} output hash mismatch: {name}')
    from importlib.metadata import version
    for package, expected_version in proto['software'].items():
        observed = torch.__version__ if package == 'torch' else version(package)
        require(observed == expected_version, f'locked {package} version mismatch: {observed} vs {expected_version}')
    report=json.loads(val.read_text())
    require(report['status']=='complete' and report['validation_only'] is True and
            report['test_ood_read'] is False and
            report['model_ids']==[f'M{i}' for i in range(8)],
            'benchmark validation status unexpected')
    m3=base/'M3_esm2_bilstm_transformer'
    require(sha(m3/'model.py')==M3_MODEL_SHA and sha(m3/'train.py')==M3_TRAIN_SHA,
            'baseline M3 classifier implementation changed')
    record=proto['models']['M3']
    ckpt=m3/'runs/seed42/best_model.pt'
    require(sha(ckpt)==BASELINE_SHA and record['checkpoint_sha256']==BASELINE_SHA,
            'frozen M3 selected checkpoint mismatch')
    cfg=json.loads((m3/'config.json').read_text())
    require(cfg['backbone_revision']==ESM_COMMIT and cfg['hidden_size']==1280,
            'baseline M3 encoder configuration changed')
    require(sha(m3/'config.json')==record['config_sha256'] and
            sha(m3/'train.py')==record['source_sha256'],
            'baseline M3 source or config differs from locked protocol')
    frames=load_development(root)  # ONLY frozen train and validation files
    require(len(frames['train'])==12152 and len(frames['validation'])==1500,
            'certified training and validation counts differ')
    return proto, cfg, ckpt, m3, frames


def load_head(m3,cfg,ckpt):
    # Ensure the local original M3 model.py is used, rather than a cached M2/M3 module.
    sys.modules.pop('model',None)
    sys.path.insert(0,str(m3))
    spec=importlib.util.spec_from_file_location('amr_m3_ft_baseline_head',m3/'train.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    head=mod.create_model(cfg).to('cuda')
    require(sum(p.numel() for p in head.parameters())==PARAMS_HEAD,
            'M3 head trainable parameter count mismatch')
    saved=torch.load(ckpt,map_location='cpu',weights_only=True)
    require(saved['epoch']==4 and saved['selection_criterion']=='validation_loss' and
            saved['backbone_commit']==ESM_COMMIT,
            'M3 baseline checkpoint provenance unexpected')
    head.load_state_dict(saved['model'],strict=True)
    return head,mod


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
    return pad_sequence([hidden[i][mask[i]].float() for i in range(len(mask))],
                        batch_first=True,padding_value=0.0)


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


def row_history(history):
    import io
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(history)
    return buffer.getvalue().encode()


def ensure_no_partials(run):
    require(not any(p.name.endswith(('.partial','.tmp')) for p in run.glob('*')),
            'orphaned partial output in fine-tuning run')


def smoke_local():
    # CPU-only functional tests. Does not read real project files.
    assert abs(schedule_lambda(0,5,100)-.2)<1e-12
    assert abs(schedule_lambda(4,5,100)-1)<1e-12
    assert abs(schedule_lambda(100,5,100))<1e-12
    assert [list(map(len,x)) for x in group_batches([[0],[1,2],[3]],2)]==[[1,2],[1]]
    probs=np.array([.9,.1,.8,.2]); labels=np.array([1,0,1,0])
    m=metrics(labels,probs)
    require(abs(m['mcc']-1)<1e-12 and abs(m['accuracy']-1)<1e-12,
            'synthetic metrics incorrect')
    # Verify exact accumulation weighting for uneven last group (2+1).
    a=torch.nn.Parameter(torch.tensor(0.1))
    x=[torch.tensor([1.,2.]),torch.tensor([3.])]
    y=[torch.tensor([1.,0.]),torch.tensor([1.])]
    for xa,ya in zip(x,y):
        (nn.functional.binary_cross_entropy_with_logits(a*xa,ya,reduction='sum')/3).backward()
    ref=torch.nn.Parameter(torch.tensor(0.1))
    nn.functional.binary_cross_entropy_with_logits(ref*torch.cat(x),torch.cat(y)).backward()
    require(torch.allclose(a.grad,ref.grad,atol=1e-7),
            'gradient accumulation sample weighting incorrect')
    # Synthetic packed BiLSTM and Transformer test using original M3 architecture.
    root_m3 = Path(__file__).resolve().parents[2] / 'training_v2' / 'M3_esm2_bilstm_transformer'
    if root_m3.is_dir():
        require(sha(root_m3/'model.py') == M3_MODEL_SHA,
                'original M3 head model source changed')
        sys.modules.pop('model',None)
        sys.path.insert(0, str(root_m3))
        spec=importlib.util.spec_from_file_location('amr_m3_lora_smoke', root_m3/'train.py')
        base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
        cfg=json.loads((root_m3/'config.json').read_text())
        head=base.create_model(cfg).cpu()
        require(sum(p.numel() for p in head.parameters())==PARAMS_HEAD,
                'M3 synthetic head parameter count mismatch')
        torch.manual_seed(78)
        xx=torch.randn(2,19,1280,requires_grad=True)
        lengths=torch.tensor([11,19],dtype=torch.long)
        head.eval()
        with torch.no_grad():
            a=head(xx, lengths)
            enlarged=torch.cat([xx, torch.full((2,7,1280),50.0)],dim=1)
            b=head(enlarged,lengths)
            require(torch.allclose(a,b,atol=3e-4,rtol=3e-4),
                    'M3 synthetic padding invariance failed')
        head.train(); head.zero_grad(set_to_none=True)
        z=nn.functional.binary_cross_entropy_with_logits(
            head(xx,lengths),torch.tensor([0.,1.]))
        z.backward()
        require(xx.grad is not None and torch.isfinite(xx.grad).all(),
                'M3 synthetic input gradient missing')
        require(head.bilstm.weight_ih_l0_reverse.grad is not None and
                head.encoder.layers[0].self_attn.in_proj_weight.grad is not None,
                'M3 synthetic BiLSTM or Transformer gradient missing')
        print('M3 original packed BiLSTM, Transformer and padding test: PASS',flush=True)
    print('M3 LORA TRAINER SYNTHETIC SMOKE: PASS',flush=True)
    print('Optimizer schedule, sample-weighted accumulation, metrics: PASS')
    print('No project files written.',flush=True)


def main(root,seed,smoke,gpu_smoke=False):
    if smoke:
        smoke_local(); return
    from transformers import AutoTokenizer,AutoModel
    from peft import LoraConfig,get_peft_model
    from peft.utils import get_peft_model_state_dict,set_peft_model_state_dict
    require(seed in (42,52,62),'seed must be from locked protocol')
    check_peft_environment()
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'GPU BF16 required')
    require(torch.cuda.get_device_properties(0).total_memory>=18*2**30,
            'GPU requires >=18 GiB')
    proto,cfg,ckpt,m3,frames=load_inputs(root,smoke)
    params=proto['training']
    run=root/f'finetuning_v2/M3_lora/seed{seed}'
    require(not (run/'training_summary.json').exists(),
            'M3 LoRA seed run already complete: cannot overwrite')
    if run.exists():ensure_no_partials(run)
    require(not any((run/f).exists() for f in ('best_model.pt','history.csv')) or
            (run/'last_checkpoint.pt').exists(),
            'orphaned run output: inspect before proceeding')
    train,val=frames['train'],frames['validation']
    seed_everything(seed)
    device='cuda'
    print(f'M3 LORA seed={seed} | training {len(train)} validation {len(val)}',flush=True)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg['backbone'],revision=ESM_COMMIT,use_fast=True)
    encoder=AutoModel.from_pretrained(cfg['backbone'],revision=ESM_COMMIT,dtype=torch.bfloat16).to(device)
    encoder.requires_grad_(False);encoder.eval()
    head,baseline_mod=load_head(m3,cfg,ckpt)
    head.eval()
    # Before any optimizer updates: same live tensors, before vs after zero LoRA adapter.
    ids=[int(train['length'].idxmin()),int(train['length'].idxmax())]
    tokens,attention,keep,lengths,targets=get_tokens(tokenizer,train,ids)
    with torch.inference_mode():
        frozen_logits=head(features(encoder(input_ids=tokens,attention_mask=attention).last_hidden_state,keep),lengths).detach().clone()
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
        after=head(features(encoder(input_ids=tokens,attention_mask=attention).last_hidden_state,keep),lengths).detach()
    max_delta=float((after-frozen_logits).abs().max())
    require(max_delta<=0.02,'LoRA adapter initialization changed logits')
    print(f'M3 LoRA zero-adapter logit parity max_delta={max_delta:.8f} PASS',flush=True)
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
            logits=head(features(hidden,keep),lengths)
            loss=nn.functional.binary_cross_entropy_with_logits(logits.float(),targets)
        require(bool(torch.isfinite(loss)),'M3 GPU smoke nonfinite loss')
        loss.backward()
        require(any(p.grad is not None and torch.isfinite(p.grad).all() and
                    bool((p.grad.abs().max()>0).item()) for p in adapt),
                'M3 GPU smoke no LoRA gradients')
        require(any(p.grad is not None and torch.isfinite(p.grad).all() and
                    bool((p.grad.abs().max()>0).item()) for p in hp),
                'M3 GPU smoke no head gradients')
        require(any('bilstm' in n and p.grad is not None and
                    torch.isfinite(p.grad).all() and bool((p.grad.abs().max()>0).item())
                    for n,p in head.named_parameters()),
                'M3 GPU smoke no bidirectional LSTM gradients')
        require(any('encoder.layers' in n and p.grad is not None and
                    torch.isfinite(p.grad).all() and bool((p.grad.abs().max()>0).item())
                    for n,p in head.named_parameters()),
                'M3 GPU smoke no head Transformer gradients')
        torch.nn.utils.clip_grad_norm_(adapt+hp,max_norm=1.0)
        optimizer.step()
        # Save/load exactly the adapter/head payload used by training, in memory.
        state={'adapter':get_peft_model_state_dict(encoder),'head':head.state_dict()}
        buff=io.BytesIO();torch.save(state,buff);buff.seek(0)
        loaded=torch.load(buff,map_location='cpu',weights_only=True)
        require(set(state['adapter'])==set(loaded['adapter']) and
                set(state['head'])==set(loaded['head']),
                'M3 GPU smoke adapter/head checkpoint roundtrip failed')
        print(f'M3 GPU TRAINING SMOKE: PASS | loss={loss.item():.5f} '
              f'| peak allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB '
              f'| peak reserved={torch.cuda.max_memory_reserved()/2**30:.2f} GiB',flush=True)
        print('LoRA gradients + head gradients + optimizer + in-memory checkpoint: PASS',flush=True)
        print('No project files written. Test/OOD not opened.',flush=True)
        return
    # Two proteins per microbatch; using frozen M3 length-bucket batching logic.
    lens=train['length'].to_numpy(dtype=np.int64)
    vlens=val['length'].to_numpy(dtype=np.int64)
    def make_train_batches():
        return baseline_mod.make_batches(np.arange(len(train)),lens,
            int(params['microbatch_size']),int(cfg['max_padded_residues_per_batch']),
            shuffle=True,rng=train_rng)
    # Exactly 6076 two-sequence microbatches for certified training split.
    n_micro=int(math.ceil(len(train)/params['microbatch_size']))
    require(n_micro==6076,'unexpected microbatch count')
    updates=int(math.ceil(n_micro/params['gradient_accumulation_steps']))
    total_steps=updates*params['max_epochs']
    warmup=int(math.ceil(total_steps*params['warmup_fraction']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step:schedule_lambda(step,warmup,total_steps))
    val_batches=baseline_mod.make_batches(np.arange(len(val)),vlens,
        int(params['microbatch_size']),int(cfg['max_padded_residues_per_batch']),
        shuffle=False,rng=train_rng)
    require(sum(map(len,val_batches))==len(val),'validation batch coverage incomplete')
    history=[]; best_epoch=0; best_loss=float('inf'); patience=0
    update_count=0; epoch_start=1; start=time.monotonic()
    if (run/'last_checkpoint.pt').exists():
        last=torch.load(run/'last_checkpoint.pt',map_location='cpu',weights_only=True)
        require(last['protocol_sha256']==PROTOCOL_SHA and
                last['baseline_sha256']==BASELINE_SHA and
                last['seed']==seed and last['backbone_commit']==ESM_COMMIT,
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
                    logits=head(features(hidden,k),ln)
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
                    logits=head(features(hidden,k),ln)
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
                'epoch':epoch,'model_id':'M3','seed':seed,
                'adapter':get_peft_model_state_dict(encoder),
                'head':head.state_dict(),
                'protocol_sha256':PROTOCOL_SHA,
                'audit_sha256':AUDIT_SHA,
                'baseline_sha256':BASELINE_SHA,
                'backbone_commit':ESM_COMMIT,
                'validation_loss':best_loss,'selection_criterion':'validation_loss'})
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
            'epoch':epoch,'model_id':'M3','seed':seed,
            'adapter':get_peft_model_state_dict(encoder),
            'head':head.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'protocol_sha256':PROTOCOL_SHA,
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
    out={'status':'complete','dataset_version':'V2','model_id':'M3','adaptation':'LoRA',
         'seed':seed,'train_rows':len(train),'validation_rows':len(val),
         'baseline_checkpoint_sha256':BASELINE_SHA,
         'protocol_sha256':PROTOCOL_SHA,'audit_sha256':AUDIT_SHA,
         'backbone_commit':ESM_COMMIT,'best_epoch':best_epoch,
         'epochs_executed':len(history),'stop_reason':stop_reason,
         'selection_criterion':'validation_loss',
         'best_validation_loss':best_loss,'best_validation_metrics':best_record,
         'best_model_sha256':sha(run/'best_model.pt'),
         'last_checkpoint_sha256':sha(run/'last_checkpoint.pt'),
         'history_sha256':sha(run/'history.csv'),
         'test_ood_accessed':False,'elapsed_seconds':time.monotonic()-start}
    write_atomic(run/'training_summary.json',json.dumps(out,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(f'M3 LORA TRAINING COMPLETE | seed {seed} | best epoch {best_epoch}',flush=True)
    print('Summary:',run/'training_summary.json',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--gpu-smoke',action='store_true')
    args=ap.parse_args()
    require(not (args.smoke and args.gpu_smoke),'choose one smoke mode')
    main(args.root,args.seed,args.smoke,args.gpu_smoke)
