#!/usr/bin/env python3
"""Exploratory ESKAPE reference-proteome inference with six selected checkpoints.

Use the original validation-locked inference implementations for model mathematics.
ESKAPE input batching is defined within each assembly, *not* across assemblies.
No training, threshold fitting, labels, classification performance or test/OOD reads.
Results are an EXPLORATORY proteome application, not independent AMR validation.

Modes:
  --self-test            Synthetic, no research files.
  --verify-inputs --root PROJECT   Verify eligible proteome inputs only; no model/checkpoint load.
  --run-all --root PROJECT  Verify eligible inputs and the original lock, then score six panels sequentially.
  --run-panel PANEL --root PROJECT  Restartable subset (same policy, same outputs).

The six per-panel frozen classifiers reuse an original BF16 ESM-2 forward pass
and the verified FP16 residue/pooled cache + FP32 heads. LoRA checkpoints are
loaded sequentially. Only original locked weights are used. This orchestration
is independent of the confirmatory held-out evaluation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

EXPECTED_E1_SOURCE_MANIFEST_SHA = '81d4938bd2a6ad74df5d7744bf74174671a560042a0502889d0ece40441afb5a'
EXPECTED_E1_QC_HASH_MANIFEST_SHA = '14ca443b0fedc8d261a035829a76407f760edb8a6509ce497eba4b1f68d99efa'
EXPECTED_ELIGIBLE_HASH_MANIFEST_SHA = '94b368c962ba319ba303a9ff61288cd74fb831b676a8b1697d4927606a2e1958'
EXPECTED_PHASE2_INFERENCE_SUMMARY_SHA = '226252c29f1d86b1cba6c6a42d1ae87bdf41c68098d52b475a8b0e2f95ffbb47'
PROJECT_DEFAULT = Path('.')
E1_REL='evaluation_v2/eskape_e1_reference_proteomes_v1'
QC_REL='evaluation_v2/eskape_e1_proteome_qc_v1'
INPUT_REL='evaluation_v2/eskape_e1_eligible_inputs_v1'
OUT_REL='evaluation_v2/eskape_e2_six_checkpoint_predictions_v1'
LOCK_REL='evaluation_v2/inference_lock_v1'
AMINO='ACDEFGHIKLMNPQRSTVWY'
MODELS=('M1','M3','M0')
REGIMES=('frozen','LoRA')
# Counts derive from the original, independently SHA-verified QC, not newly fitted data.
EXPECTED_COUNTS={
    'E_faecium_DO':2705,
    'S_aureus_NCTC8325':2737,
    'K_pneumoniae_MGH78578':5190,
    'A_baumannii_ATCC19606':3621,
    'P_aeruginosa_PAO1':5491,
    'E_cloacae_ATCC13047':5111,
}
EXPECTED_ASSEMBLIES={
    'E_faecium_DO':'GCF_000174395.2',
    'S_aureus_NCTC8325':'GCF_000013425.1',
    'K_pneumoniae_MGH78578':'GCF_000016305.1',
    'A_baumannii_ATCC19606':'GCF_009035845.1',
    'P_aeruginosa_PAO1':'GCF_000006765.1',
    'E_cloacae_ATCC13047':'GCF_000025565.1',
}


def require(ok,why):
    if not bool(ok):
        raise RuntimeError('STOP: '+why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write_json_atomic(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    require(not path.exists() and not tmp.exists(),'Existing output: '+str(path))
    with tmp.open('x',encoding='utf8') as f:
        json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read_fasta(path):
    header=None;lines=[]
    with Path(path).open('r',encoding='ascii') as f:
        for raw in f:
            line=raw.strip()
            if not line:continue
            if line.startswith('>'):
                if header is not None:yield header,''.join(lines)
                header=line[1:]
                require(bool(header) and not any(c.isspace() for c in header),
                        'Non-atomic FASTA key: '+repr(header))
                lines=[]
            else:
                require(header is not None,'Sequence before FASTA identifier')
                lines.append(line)
        if header is not None:yield header,''.join(lines)


def checked_boolean(series,field):
    v=series.astype(str).str.lower()
    require(bool(v.isin(('true','false')).all()),'Invalid boolean column: '+field)
    return v.eq('true')


def verify_e1(root):
    """Check entire immutable eligible-input package, not just filenames."""
    root=Path(root);e1=root/E1_REL;qc=root/QC_REL;inputs=root/INPUT_REL
    require(root.is_dir(),'Project root not mounted')
    checks=[
        (e1/'E1_REFERENCE_MANIFEST.json',EXPECTED_E1_SOURCE_MANIFEST_SHA),
        (qc/'OUTPUT_HASHES.json',EXPECTED_E1_QC_HASH_MANIFEST_SHA),
        (inputs/'OUTPUT_HASHES.json',EXPECTED_ELIGIBLE_HASH_MANIFEST_SHA),
    ]
    for p,expected in checks:
        require(p.is_file() and sha(p)==expected,'reference-proteome source, QC, or input manifest mismatch: '+str(p))
    for folder in (qc,inputs):
        manifest=read_json(folder/'OUTPUT_HASHES.json')
        for rel,digest in manifest.items():
            p=folder/rel
            require(p.is_file() and sha(p)==digest,'Certified reference-proteome output changed: '+str(p))
    refs=read_json(e1/'E1_REFERENCE_MANIFEST.json')['references']
    refmap={r['panel_id']:r for r in refs}
    require(set(refmap)==set(EXPECTED_COUNTS),'Six original panel records changed')
    for panel,r in refmap.items():
        require(r['assembly_accession']==EXPECTED_ASSEMBLIES[panel],
                'Reference assembly changed: '+panel)
    er=read_json(inputs/'INPUT_ELIGIBILITY_RECORD.json')
    require(er.get('n_eligible')==24855 and er.get('n_excluded')==310 and er.get('n_raw')==25165,
            'reference-proteome eligibility totals changed')
    require(er.get('homology_proximity')=='not_yet_assessed' or
            er.get('homology_proximity_status')=='NOT ASSESSED; no claims of distant-homology novelty permitted',
            'Unexpected reference-proteome homology proximity statement')
    table=pd.read_csv(inputs/'PROTEIN_ELIGIBILITY_AND_OVERLAP.csv',dtype=str,keep_default_na=False)
    require(len(table)==25165 and table['protein_key'].is_unique,'Invalid reference-proteome protein catalog')
    flag=checked_boolean(table['inference_eligible'],'inference_eligible')
    require(int(flag.sum())==24855,'Eligible catalog count changed')
    for c in ['exact_overlap_train','exact_overlap_validation','exact_overlap_test','exact_overlap_ood',
              'exact_overlap_any','exploratory_nonexact_subset']:
        table[c]=checked_boolean(table[c],c)
    require(int(table['exact_overlap_any'].sum())==11 and
            int(table['exact_overlap_train'].sum())==9 and
            int(table['exact_overlap_test'].sum())==2 and
            int(table['exact_overlap_validation'].sum())==0 and
            int(table['exact_overlap_ood'].sum())==0,
            'reference-proteome overlap accounting changed')
    require(not table['exact_overlap_any'].ne(table[[
        'exact_overlap_train','exact_overlap_validation','exact_overlap_test','exact_overlap_ood']].any(axis=1)).any(),
            'Aggregate overlap does not match partition flags')
    require(not (table['exploratory_nonexact_subset'] & (~flag | table['exact_overlap_any'])).any(),
            'Nonexact subset flag inconsistent')
    require(int(table['exploratory_nonexact_subset'].sum())==24844,
            'Nonexact eligible set count changed')
    require(int(table['length_aa'].astype(int).gt(1014).sum())>=296,
            'Length eligibility unexpectedly changed')
    print('REFERENCE-PROTEOME SOURCE, QC, ELIGIBILITY AND OVERLAP HASHES: PASS',flush=True)
    return inputs,table,refmap,flag


def load_panel(inputs,table,refmap,panel):
    require(panel in EXPECTED_COUNTS,'Unknown panel: '+str(panel))
    selected=table.loc[(table.panel_id==panel)&(table.inference_eligible.str.lower()=='true')].copy()
    selected=selected.sort_values('protein_key',kind='stable').reset_index(drop=True)
    n=EXPECTED_COUNTS[panel]
    require(len(selected)==n and selected['protein_key'].is_unique,
            'Eligible input count differs: '+panel)
    fasta_path=inputs/'FASTA'/f'{panel}_eligible.faa'
    rows=list(read_fasta(fasta_path))
    require(len(rows)==n,'FASTA count mismatch: '+panel)
    ids=[h for h,_ in rows]
    seq=[s for _,s in rows]
    require(ids==selected['protein_key'].tolist(),'FASTA/catalog order or identifiers differ: '+panel)
    lengths=selected['length_aa'].astype(int).to_numpy()
    require(all(1<=len(s)<=1014 and set(s)<=set(AMINO) for s in seq),
            'Sequence violates original no-truncation input policy: '+panel)
    require(np.array_equal(lengths,np.asarray([len(s) for s in seq],dtype=int)),
            'Eligible lengths disagree: '+panel)
    require([sha_bytes(s.encode('ascii')) for s in seq]==selected['sequence_sha256'].tolist(),
            'Eligible sequence hash differs: '+panel)
    require(int(selected['exact_overlap_any'].sum())==(11 if panel=='A_baumannii_ATCC19606' else 0),
            'Eligible exact overlap changed: '+panel)
    frame=pd.DataFrame({
        'protein_id':ids, 'sequence':seq,'length':lengths,
    })
    require(refmap[panel]['assembly_accession']==EXPECTED_ASSEMBLIES[panel],
            'Assembly accession changed: '+panel)
    return frame, selected, fasta_path


def synthetic_test():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'small.faa'
        p.write_text('>a\nACD\n>bb\nW\nYK\n',encoding='ascii')
        require(list(read_fasta(p))==[('a','ACD'),('bb','WYK')],
                'Synthetic multiline FASTA parse mismatch')
        q=Path(tmp)/'p.csv'
        sample=pd.DataFrame({'protein_key':['a','bb'],
              'probability_amr':[.2,.8],'prediction_at_0_5':[0,1]})
        sample.to_csv(q,index=False,float_format='%.17g')
        got=pd.read_csv(q)
        require(got.protein_key.tolist()==['a','bb'] and
                got.prediction_at_0_5.tolist()==[0,1],
                'Synthetic result alignment mismatch')
    print('SYNTHETIC FASTA, ROW-ALIGNMENT AND SCORE SELF-TEST: PASS')
    print('No project data/checkpoints or test/OOD files read.')


def verify_original_inference(root):
    """Use archived locked code as a library; never modify or copy it into model dirs."""
    import torch
    lock=root/LOCK_REL
    require(lock.is_dir(),'Original inference code lock not found')
    sys.path.insert(0,str(lock))
    import hyposeq_locked_holdout_v1 as hold
    import hyposeq_inference_validation_v1 as base
    hold.verify_lock(root)  # Exact locked source/validation policy and sha attestation
    inv,proto=base.verified_preflight(root)
    base.verify_checkpoint_structures(root,inv)
    base.runtime_guard(proto)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'CUDA BF16 required; switch to NVIDIA L4')
    require(sha(root/'evaluation_v2/phase2_holdout_predictions_v1/FINAL_INFERENCE_MANIFEST.json')
            ==EXPECTED_PHASE2_INFERENCE_SUMMARY_SHA,'completed held-out inference summary checksum changed')
    candidates={(r['model'],r['regime']):r for r in inv['candidates']}
    require(set(candidates)=={(m,reg) for m in MODELS for reg in REGIMES},
            'Six checkpoint identity mismatch')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    print('LOCKED INFERENCE SOURCE + SIX CHECKPOINTS: PASS (reused unchanged)',flush=True)
    return hold,inv,proto,candidates


def result_root(root,panel,model,regime):
    return root/OUT_REL/panel/f'{model}_{regime}'


def check_result(root,panel,model,regime,df,selected,fasta_sha,checkpoint_sha):
    p=result_root(root,panel,model,regime)
    require(not p.with_name(p.name+'.partial').exists(),'Incomplete prior output; preserve/inspect: '+str(p))
    if not p.exists():return False
    record=read_json(p/'RESULT.json')
    require(record['schema']=='hyposeq_eskape_e2_exploratory_prediction_v1'
            and record['status']=='complete' and record['model']==model
            and record['regime']==regime and record['panel_id']==panel
            and record['assembly_accession']==EXPECTED_ASSEMBLIES[panel]
            and record['n']==len(df) and record['threshold']==0.5
            and record['input_fasta_sha256']==fasta_sha
            and record['checkpoint_sha256']==checkpoint_sha
            and record['runner_sha256']==sha(Path(__file__))
            and record['inference_lock_manifest_sha256']==sha(root/LOCK_REL/'INFERENCE_LOCK_MANIFEST.json'),
            'Existing model output provenance mismatch: '+str(p))
    csv=p/'predictions.csv'
    require(csv.is_file() and sha(csv)==record['predictions_sha256'],
            'Existing model output hash mismatch: '+str(p))
    got=pd.read_csv(csv,dtype={'protein_key':str})
    require(len(got)==len(df) and got['protein_key'].tolist()==df['protein_id'].tolist(),
            'Existing model output ID mismatch: '+str(p))
    require(got['prediction_at_0_5'].isin([0,1]).all() and
            got['probability_amr'].between(0,1).all() and
            got['probability_amr'].notna().all(),
            'Invalid existing model probabilities: '+str(p))
    # CSV values are rounded at 10 significant digits. Fail only on real mismatches.
    require(np.array_equal(got['prediction_at_0_5'].to_numpy(dtype=int),
                   (got['probability_amr'].to_numpy(dtype=float)>=.5).astype(int)),
            'Existing threshold flags differ: '+str(p))
    require(int(got['prediction_at_0_5'].sum())==record['n_flagged_positive'],
            'Existing prediction summary differs: '+str(p))
    print('SKIP VERIFIED:',panel,model,regime,flush=True)
    return True


def commit_result(root,panel,model,regime,df,selected,fasta_sha,candidate,probabilities):
    where=result_root(root,panel,model,regime)
    temporary=where.with_name(where.name+'.partial')
    require(not where.exists() and not temporary.exists(),'Existing model output: '+str(where))
    probs=np.asarray(probabilities,dtype=np.float64)
    require(probs.shape==(len(df),) and np.isfinite(probs).all()
            and ((probs>=0)&(probs<=1)).all(),'Invalid model output probabilities')
    flags=(probs>=.5).astype(np.int8)
    temporary.mkdir(parents=True,exist_ok=False)
    export=pd.DataFrame({
        'panel_id':panel,
        'assembly_accession':EXPECTED_ASSEMBLIES[panel],
        'protein_key':df['protein_id'],
        'original_protein_id':selected['original_protein_id'].astype(str),
        'sequence_sha256':selected['sequence_sha256'].astype(str),
        'length_aa':df['length'],
        'exact_overlap_train':selected['exact_overlap_train'].astype(bool),
        'exact_overlap_validation':selected['exact_overlap_validation'].astype(bool),
        'exact_overlap_test':selected['exact_overlap_test'].astype(bool),
        'exact_overlap_ood':selected['exact_overlap_ood'].astype(bool),
        'exact_overlap_any':selected['exact_overlap_any'].astype(bool),
        'probability_amr':probs,
        'prediction_at_0_5':flags,
    })
    export.to_csv(temporary/'predictions.csv',index=False,float_format='%.17g')
    r={
        'schema':'hyposeq_eskape_e2_exploratory_prediction_v1',
        'status':'complete','exploratory_unlabeled_application':True,
        'panel_id':panel,'assembly_accession':EXPECTED_ASSEMBLIES[panel],
        'model':model,'regime':regime,'n':len(df),'n_flagged_positive':int(flags.sum()),
        'fraction_flagged_positive':float(flags.mean()),
        'threshold':0.5,'checkpoint_sha256':candidate['checkpoint_sha256'],
        'selected_seed':candidate['seed'],'selected_epoch':candidate['selected_epoch'],
        'input_fasta_sha256':fasta_sha,
        'E1_eligibility_hash_manifest_sha256':EXPECTED_ELIGIBLE_HASH_MANIFEST_SHA,
        'runner_sha256':sha(Path(__file__)),
        'inference_lock_manifest_sha256':sha(root/LOCK_REL/'INFERENCE_LOCK_MANIFEST.json'),
        'predictions_sha256':sha(temporary/'predictions.csv'),
        'batch_policy':('ESM2 frozen original 4 sequences/2048 padded tokens by stable length; '
                        'LoRA length-sort pairs; each RefSeq assembly independently'),
        'caution':'Scores are not externally calibrated and no complete AMR truth labels exist. '
                  'Counts/fractions are descriptive, not accuracy, sensitivity, specificity or prevalence.',
        'created_utc':datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(temporary/'RESULT.json',r)
    os.replace(temporary,where)
    print(f'RESULT {panel}/{model}_{regime}: {int(flags.sum())}/{len(flags)} flagged, '
          f'CSV SHA256={r["predictions_sha256"]}',flush=True)
    return r


def save_panel_matrix(root,panel,df,selected,candidates):
    import pandas as pd
    base=root/OUT_REL/panel
    path=base/'SIX_MODEL_SCORES.csv'
    record_path=base/'PANEL_MANIFEST.json'
    require(not path.with_name(path.name+'.partial').exists(),
            'Partial panel matrix exists: '+str(path))
    if path.exists() or record_path.exists():
        require(path.exists() and record_path.exists(), 'Incomplete prior panel summary: '+panel)
        r=read_json(record_path)
        require(sha(path)==r['matrix_sha256'] and len(r['models'])==6,
                'Existing panel summary hash mismatch: '+panel)
        print('SKIP VERIFIED PANEL MATRIX:',panel,flush=True)
        return r
    matrix=pd.DataFrame({
        'panel_id':panel,'assembly_accession':EXPECTED_ASSEMBLIES[panel],
        'protein_key':df['protein_id'], 'original_protein_id':selected['original_protein_id'].astype(str),
        'sequence_sha256':selected['sequence_sha256'].astype(str),
        'length_aa':df['length'],
        'exact_overlap_any':selected['exact_overlap_any'].astype(bool),
    })
    models=[]
    for model in MODELS:
        for regime in REGIMES:
            res=result_root(root,panel,model,regime)
            rec=read_json(res/'RESULT.json')
            require(sha(res/'predictions.csv')==rec['predictions_sha256'],
                    'Model prediction hash changed: '+str(res))
            dfres=pd.read_csv(res/'predictions.csv',dtype={'protein_key':str})
            require(dfres['protein_key'].tolist()==df['protein_id'].tolist(),
                    'Matrix row alignment mismatch: '+panel+'/'+model+'_'+regime)
            matrix[f'{model}_{regime}_score']=dfres['probability_amr'].to_numpy(dtype=float)
            matrix[f'{model}_{regime}_flag']=dfres['prediction_at_0_5'].to_numpy(dtype=int)
            models.append({'model':model,'regime':regime,'predictions_csv_sha256':rec['predictions_sha256'],
                           'checkpoint_sha256':candidates[model,regime]['checkpoint_sha256'],
                           'n_flagged_positive':rec['n_flagged_positive']})
    tmp=path.with_name(path.name+'.partial')
    matrix.to_csv(tmp,index=False,float_format='%.17g')
    os.replace(tmp,path)
    write_json_atomic(record_path,{
        'schema':'hyposeq_eskape_e2_six_model_panel_manifest_v1','status':'complete',
        'panel_id':panel,'assembly_accession':EXPECTED_ASSEMBLIES[panel],
        'n':len(df),'matrix_sha256':sha(path),'models':models,
        'no_ground_truth_or_accuracy_claim':True,
    })
    print('SIX-MODEL MATRIX:',panel,sha(path),flush=True)
    return read_json(record_path)


def run(root,panels):
    root=Path(root)
    inputs,table,refmap,_=verify_e1(root)
    hold,inv,proto,candidates=verify_original_inference(root)
    out=root/OUT_REL
    require(out.is_dir() or not out.exists(),'Invalid ESKAPE inference destination')
    out.mkdir(parents=True,exist_ok=True)
    scriptcopy=out/'E2_RUNNER_SOURCE.py'
    if scriptcopy.exists():
        require(sha(scriptcopy)==sha(Path(__file__)),'Original ESKAPE runner source differs')
    else:
        shutil.copyfile(Path(__file__),scriptcopy)
        require(sha(scriptcopy)==sha(Path(__file__)),'ESKAPE runner source copy mismatch')
    for panel in panels:
        df,selected,source_fasta=load_panel(inputs,table,refmap,panel)
        fasta_sha=sha(source_fasta)
        print('\n'+'='*73+'\nSTART PANEL:',panel,
              EXPECTED_ASSEMBLIES[panel],len(df),'eligible proteins',flush=True)
        done={}
        for model in MODELS:
            for regime in REGIMES:
                done[model,regime]=check_result(root,panel,model,regime,df,selected,fasta_sha,
                                       candidates[model,regime]['checkpoint_sha256'])
        pending_frozen=[m for m in MODELS if not done[m,'frozen']]
        if pending_frozen:
            scratch,offset,pooled=hold.frozen_shared_features(root,df,'eskape_'+panel)
            all_frozen_complete=False
            try:
                for model in pending_frozen:
                    p=hold.frozen_score(root,model,candidates[model,'frozen'],df,scratch,offset,pooled)
                    commit_result(root,panel,model,'frozen',df,selected,fasta_sha,
                                  candidates[model,'frozen'],p)
                all_frozen_complete=True
            finally:
                if all_frozen_complete:
                    del pooled,offset
                    try:scratch.unlink()
                    except OSError:
                        print('WARNING: Frozen scratch remains:',scratch,flush=True)
        for model in MODELS:
            if not done[model,'LoRA']:
                p=hold.lora_score(root,inv,proto,model,df)
                commit_result(root,panel,model,'LoRA',df,selected,fasta_sha,candidates[model,'LoRA'],p)
        save_panel_matrix(root,panel,df,selected,candidates)
        print('PANEL COMPLETE:',panel,flush=True)
    # Global summary only when all six panels have complete matrices.
    if all((out/p/'PANEL_MANIFEST.json').is_file() for p in EXPECTED_COUNTS):
        summary=out/'GLOBAL_E2_MANIFEST.json'
        if summary.exists():
            old=read_json(summary)
            require(old['source_runner_sha256']==sha(Path(__file__)),
                    'Existing global manifest source differs')
            print('GLOBAL MANIFEST ALREADY SAVED:',sha(summary),flush=True)
        else:
            matrix_records=[]
            for panel in EXPECTED_COUNTS:
                rec=read_json(out/panel/'PANEL_MANIFEST.json')
                require(rec['panel_id']==panel and rec['n']==EXPECTED_COUNTS[panel]
                        and sha(out/panel/'SIX_MODEL_SCORES.csv')==rec['matrix_sha256'],
                        'Global panel manifest mismatch: '+panel)
                matrix_records.append({
                    'panel_id':panel,'assembly':EXPECTED_ASSEMBLIES[panel],
                    'panel_manifest_sha256':sha(out/panel/'PANEL_MANIFEST.json'),
                    'six_model_matrix_sha256':rec['matrix_sha256'],
                    'n_eligible':rec['n'],
                })
            write_json_atomic(summary,{
                'schema':'hyposeq_eskape_e2_global_exploratory_inference_v1',
                'status':'complete','n_assemblies':6,'n_eligible_proteins':24855,
                'model_checkpoints':6,'model_proteome_evaluations':36,
                'source_runner_sha256':sha(Path(__file__)),
                'source_E1_eligibility_hash_manifest_sha256':EXPECTED_ELIGIBLE_HASH_MANIFEST_SHA,
                'phase2_lock_manifest_sha256':sha(root/LOCK_REL/'INFERENCE_LOCK_MANIFEST.json'),
                'panels':matrix_records,
                'limits':'Unlabeled exploratory application. No accuracy, precision, recall, specificity, '
                         'resistance prevalence or novel family claims. Homology-proximity and functional annotation pending.',
            })
            print('GLOBAL ESKAPE INFERENCE COMPLETE: 36/36 evaluated. Manifest SHA256:',sha(summary),flush=True)
    print('COMPLETED REQUESTED PANELS:',len(panels),'; original study data/checkpoints unchanged.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    opts=ap.add_mutually_exclusive_group(required=True)
    opts.add_argument('--self-test',action='store_true')
    opts.add_argument('--verify-inputs',action='store_true')
    opts.add_argument('--run-all',action='store_true')
    opts.add_argument('--run-panel',choices=list(EXPECTED_COUNTS))
    ap.add_argument('--root',type=Path,default=PROJECT_DEFAULT)
    a=ap.parse_args()
    if a.self_test:synthetic_test()
    elif a.verify_inputs:
        inputs,table,refs,flag=verify_e1(a.root)
        for panel in EXPECTED_COUNTS:
            df,selected,path=load_panel(inputs,table,refs,panel)
            print('VERIFIED ESKAPE ELIGIBLE FASTA:',panel,len(df),sha(path),flush=True)
        print('ELIGIBLE PROTEOME INPUTS: 24,855 CERTIFIED. GPU INFERENCE NOT STARTED.',flush=True)
    else:
        run(a.root,list(EXPECTED_COUNTS) if a.run_all else [a.run_panel])

if __name__=='__main__':main()
