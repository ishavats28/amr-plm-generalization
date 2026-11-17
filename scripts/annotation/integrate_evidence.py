#!/usr/bin/env python3
"""HypoSeq 552: lossless, descriptive evidence integration; no AMR prioritization.

Inputs (existing, read-only): frozen candidate export, restored Swiss-Prot DIAMOND
results, merged InterProScan 6 web results and their provenance manifests.
Outputs: a new, immutable, versioned 03_evidence_integration directory.

Only Python standard library is required.  No model inference, remote requests,
threshold tuning, functional classification, or modification of source files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

EXPECTED_CANDIDATE_SHA = '796fa564a0c4574f7bd251e4eab5461b539635c8cf2306cb77808f533dc0577f'
EXPECTED_INTERPRO_SHA = '7bf90dd05502aa796c24e8a3556e537f6a861f4233f37885eb394cc3ab41ea6f'
BASE_REL = Path('evaluation_v2/eskape_552_characterization_v1')
IP_REL = Path('02_interproscan/processed')
OUT_REL = Path('03_evidence_integration')

class ValidationError(RuntimeError):
    pass

def need(condition, message):
    if not condition:
        raise ValidationError(message)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path, sep=','):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=sep))

def read_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def fasta_records(data):
    result = {}
    key, parts = None, []
    for line in data.decode('ascii').splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if key is not None:
                need(key not in result, 'Duplicate FASTA ID: ' + key)
                result[key] = ''.join(parts)
            key = line[1:].split()[0]
            parts = []
        else:
            need(key is not None, 'Malformed FASTA')
            parts.append(line)
    if key is not None:
        need(key not in result, 'Duplicate FASTA ID: ' + key)
        result[key] = ''.join(parts)
    return result

def unique_index(rows, key, label):
    d = {}
    for row in rows:
        k = row[key]
        need(k not in d, f'Duplicate {label} ID: {k}')
        d[k] = row
    return d

def load_and_verify(root):
    base = root / BASE_REL
    provenance = base / '00_provenance'
    diamond = base / '01_blastp'
    ip = base / IP_REL
    zip_path = provenance / 'HypoSeq_M3_Frozen_552_Hypothetical_Candidates.zip'
    need(zip_path.is_file(), f'Missing frozen ZIP: {zip_path}')
    need(sha_file(zip_path) == EXPECTED_CANDIDATE_SHA, 'Frozen candidate ZIP SHA-256 changed')
    with zipfile.ZipFile(zip_path) as z:
        need(z.testzip() is None, 'Frozen candidate ZIP damaged')
        metadata_bytes = z.read('M3_FROZEN_552_CANDIDATES.csv')
        fasta_bytes = z.read('M3_FROZEN_552_CANDIDATES.faa')
        candidate_manifest = json.loads(z.read('EXPORT_MANIFEST.json'))
    need(sha_bytes(metadata_bytes) == candidate_manifest['candidate_csv_sha256'], 'Candidate CSV digest invalid')
    need(sha_bytes(fasta_bytes) == candidate_manifest['candidate_fasta_sha256'], 'Candidate FASTA digest invalid')
    candidates = list(csv.DictReader(io.StringIO(metadata_bytes.decode('utf-8'))))
    cidx = unique_index(candidates, 'protein_key', 'candidate')
    sequences = fasta_records(fasta_bytes)
    need(len(candidates) == len(sequences) == 552, 'Candidate count not 552')
    need(set(cidx) == set(sequences), 'Candidate FASTA / CSV ID mismatch')
    for row in candidates:
        k = row['protein_key']; seq=sequences[k]
        need(row['M3_frozen_flag'] == '1', f'M3 flag differs: {k}')
        need(row['amrf_any_hit'].lower() == 'false' and row['exact_overlap_any'].lower() == 'false', f'Candidate filter invalid: {k}')
        need('hypothetical protein' in row['source_header'].lower(), f'Candidate source label differs: {k}')
        need(int(row['length_aa']) == len(seq), f'Candidate sequence length differs: {k}')
        need(sha_bytes(seq.encode('ascii')) == row['sequence_sha256'], f'Candidate sequence digest differs: {k}')
    need(candidate_manifest['candidate_count'] == 552, 'Candidate manifest count invalid')
    if (provenance / 'M3_FROZEN_552_CANDIDATES.csv').exists():
        need(sha_file(provenance / 'M3_FROZEN_552_CANDIDATES.csv') == sha_bytes(metadata_bytes), 'Drive candidate CSV mismatch')
    if (provenance / 'M3_FROZEN_552_CANDIDATES.faa').exists():
        need(sha_file(provenance / 'M3_FROZEN_552_CANDIDATES.faa') == sha_bytes(fasta_bytes), 'Drive candidate FASTA mismatch')

    diamond_manifest_path = diamond / 'DIAMOND_RUN_MANIFEST.json'
    diamond_hash_path = diamond / 'OUTPUT_HASHES.json'
    need(diamond_manifest_path.is_file() and diamond_hash_path.is_file(), 'Restored DIAMOND provenance not found')
    diamond_manifest = read_json(diamond_manifest_path)
    diamond_hashes = read_json(diamond_hash_path)
    need(diamond_manifest.get('database', {}).get('release') == '2026_04', 'Unexpected Swiss-Prot release')
    need('Swiss-Prot' in diamond_manifest.get('database', {}).get('name', ''), 'Unexpected DIAMOND database')
    need(diamond_manifest.get('input_fasta_sha256') == sha_bytes(fasta_bytes), 'DIAMOND input FASTA mismatch')
    cmd = diamond_manifest.get('commands', {}).get('blastp', [])
    need('--very-sensitive' in cmd, 'Unexpected DIAMOND sensitivity mode')
    need('--evalue' in cmd and cmd[cmd.index('--evalue') + 1] == '1e-5', 'Unexpected DIAMOND E-value')
    need('--max-target-seqs' in cmd and cmd[cmd.index('--max-target-seqs')+1] == '25', 'Unexpected DIAMOND max targets')
    diamond_required = [
        'DIAMOND_ALL_ALIGNMENTS.tsv','DIAMOND_BEST_PER_QUERY.csv',
        'DIAMOND_QUERY_SUMMARY.csv', 'DIAMOND_NO_HITS.csv',
        'DIAMOND_RUN_MANIFEST.json','diamond_blastp.log','diamond_makedb.log'
    ]
    for name in diamond_required:
        path = diamond / name
        need(path.is_file(), f'Missing restored DIAMOND file: {name}')
        expect = diamond_hashes.get('01_blastp/'+name)
        need(expect and sha_file(path) == expect, f'Invalid DIAMOND checksum: {name}')
    alignments = read_csv(diamond/'DIAMOND_ALL_ALIGNMENTS.tsv', '\t')
    summary_rows = read_csv(diamond/'DIAMOND_QUERY_SUMMARY.csv')
    best_rows = read_csv(diamond/'DIAMOND_BEST_PER_QUERY.csv')
    nohit_rows = read_csv(diamond/'DIAMOND_NO_HITS.csv')
    didx = unique_index(summary_rows, 'protein_key', 'DIAMOND summary')
    bidx = unique_index(best_rows, 'qseqid', 'DIAMOND best')
    noidx = unique_index(nohit_rows, 'protein_key', 'DIAMOND no-hit')
    need(len(didx) == 552 and set(didx) == set(cidx), 'DIAMOND query inventory incomplete')
    need(set(noidx).isdisjoint(set(bidx)), 'DIAMOND best/no-hit overlap')
    need(set(noidx) | set(bidx) == set(cidx), 'DIAMOND best/no-hit partition invalid')
    grouped = defaultdict(list)
    for row in alignments:
        k = row['qseqid']
        need(k in cidx, 'Unknown DIAMOND alignment query: '+k)
        need(float(row['evalue']) <= 1e-5, 'DIAMOND alignment exceeds E-value limit: '+k)
        need(float(row['qlen']) == int(cidx[k]['length_aa']), 'DIAMOND query length differs: '+k)
        need(row['uniprot_accession'], 'DIAMOND hit missing UniProt accession: '+k)
        grouped[k].append(row)
    for k in cidx:
        ds = didx[k]
        n = len(grouped.get(k, []))
        need(n == int(ds['n_alignment_rows']), 'DIAMOND row count differs: '+k)
        need(bool(n) == (ds['has_diamond_hit'] == '1'), 'DIAMOND has-hit flag differs: '+k)
        need((k in noidx) == (not bool(n)), 'DIAMOND no-hit inventory differs: '+k)
        need(int(ds['n_distinct_swissprot_subjects']) == len(set(v['sseqid'] for v in grouped.get(k, []))), 'DIAMOND subject counts differ: '+k)
        need(ds['panel_id'] == cidx[k]['panel_id'], 'DIAMOND panel differs: '+k)
        need(int(ds['length_aa']) == int(cidx[k]['length_aa']), 'DIAMOND length differs: '+k)
        need(abs(float(ds['M3_frozen_score'])-float(cidx[k]['M3_frozen_score'])) < 1e-12, 'DIAMOND score differs: '+k)
        if k in bidx:
            need(any(all(r[col] == bidx[k][col] for col in bidx[k]) for r in grouped[k]), 'DIAMOND best row not in full alignments: '+k)

    combined_path = ip/'HypoSeq_552_InterProScan_Combined_v1.json'
    ip_manifest_path = ip/'HypoSeq_552_InterProScan_Merge_Manifest_v1.json'
    ip_summary_path = ip/'HypoSeq_552_InterProScan_Protein_Summary_v1.csv'
    ip_delivery_path = base/'02_interproscan/DELIVERY_MANIFEST_v1.json'
    need(all(p.is_file() for p in (combined_path, ip_manifest_path, ip_summary_path, ip_delivery_path)), 'Missing saved InterProScan inputs')
    ip_manifest = read_json(ip_manifest_path)
    ip_delivery = read_json(ip_delivery_path)
    need(sha_file(combined_path) == EXPECTED_INTERPRO_SHA, 'Combined InterProScan checksum differs')
    for path in (combined_path, ip_manifest_path, ip_summary_path):
        rel = path.relative_to(root).as_posix()
        need(ip_delivery['file_sha256'].get(rel) == sha_file(path), 'InterProScan delivery checksum differs: '+rel)
    need(ip_delivery.get('reference_candidate_sha256') == EXPECTED_CANDIDATE_SHA, 'InterProScan candidate source differs')
    need(ip_manifest.get('status') == 'PASS', 'InterProScan merge was not approved')
    obj = read_json(combined_path)
    need(obj.get('interproscan-version') == '6.0.3.0' and obj.get('interpro-version') == '111.0', 'Unexpected InterProScan/InterPro version')
    results = obj['results']
    need(len(results) == 552, 'Combined InterProScan count differs')
    ipidx = {}
    for r in results:
        xref = r.get('xref', [])
        need(len(xref) == 1 and xref[0].get('id') == xref[0].get('name'), 'Ambiguous InterProScan xref')
        k = xref[0]['id']
        need(k in cidx and k not in ipidx, 'Missing or duplicate InterProScan ID: '+k)
        seq = r['sequence']
        need(seq == sequences[k], 'InterProScan full sequence mismatch: '+k)
        need(hashlib.md5(seq.encode('ascii')).hexdigest().lower() == r['md5'].lower(), 'InterProScan sequence MD5 mismatch: '+k)
        ipidx[k] = r
    need(set(ipidx) == set(cidx), 'InterProScan missing IDs')
    ip_summary = unique_index(read_csv(ip_summary_path), 'protein_key', 'InterProScan summary')
    need(set(ip_summary) == set(cidx), 'InterProScan summary inventory mismatch')
    for k, r in ipidx.items():
        s = ip_summary[k]
        matches = r.get('matches', [])
        nentry = len({str(m.get('signature',{}).get('entry',{}).get('accession')) for m in matches if isinstance(m.get('signature',{}).get('entry'),dict) and m['signature']['entry'].get('accession')})
        need(int(s['match_objects']) == len(matches), 'InterProScan match count differs: '+k)
        need(int(s['has_any_match']) == int(bool(matches)), 'InterProScan match flag differs: '+k)
        need(int(s['has_interpro_entry']) == int(bool(nentry)), 'InterProScan integrated entry flag differs: '+k)
        need(int(s['distinct_interpro_entries']) == nentry, 'InterProScan entry count differs: '+k)
        need(int(s['length_aa']) == int(cidx[k]['length_aa']), 'InterProScan summary sequence length differs: '+k)
    return candidates, cidx, sequences, grouped, didx, bidx, ipidx, ip_summary, candidate_manifest, diamond_manifest, ip_manifest, {
        'frozen_candidate_zip': sha_file(zip_path),
        'diamond_all_alignments': sha_file(diamond/'DIAMOND_ALL_ALIGNMENTS.tsv'),
        'diamond_query_summary': sha_file(diamond/'DIAMOND_QUERY_SUMMARY.csv'),
        'diamond_best_per_query': sha_file(diamond/'DIAMOND_BEST_PER_QUERY.csv'),
        'diamond_run_manifest': sha_file(diamond_manifest_path),
        'interproscan_combined_json': sha_file(combined_path),
        'interproscan_summary_csv': sha_file(ip_summary_path),
        'interproscan_merge_manifest': sha_file(ip_manifest_path),
    }

def coverage_label(has_d, has_i):
    if has_d and has_i: return 'diamond_and_interpro_entry'
    if has_d: return 'diamond_only'
    if has_i: return 'interpro_entry_only'
    return 'neither_diamond_nor_interpro_entry'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Original study root')
    parser.add_argument('--review-zip', type=Path, default=Path('integrated_evidence_review.zip'))
    parser.add_argument('--verify-only', action='store_true')
    a = parser.parse_args()
    root = a.root.resolve()
    need(root.is_dir(), 'Research root does not exist')
    (candidates, cidx, sequences, grouped, didx, bidx, ipidx, ip_summary,
     candidate_manifest, diamond_manifest, ip_manifest, sources) = load_and_verify(root)

    counters = Counter()
    complete = []
    summaries = []
    for c in candidates:
        k = c['protein_key']; d = didx[k]; s = ip_summary[k]
        has_d = bool(grouped.get(k))
        has_i = s['has_interpro_entry'] == '1'
        label = coverage_label(has_d, has_i)
        counters[label] += 1
        top = bidx.get(k, {})
        summaries.append({
            'protein_key':k, 'panel_id':c['panel_id'], 'original_protein_id':c['original_protein_id'],
            'length_aa':c['length_aa'], 'M3_frozen_score':c['M3_frozen_score'],
            'number_models_flagged':c['number_models_flagged'],
            'amrf_any_hit':c['amrf_any_hit'], 'exact_overlap_any':c['exact_overlap_any'],
            'diamond_has_hit': int(has_d), 'diamond_alignment_rows':len(grouped.get(k,[])),
            'diamond_distinct_subjects':d['n_distinct_swissprot_subjects'],
            'diamond_best_accession':top.get('uniprot_accession',''),
            'diamond_best_subject':top.get('sseqid',''),
            'diamond_best_title':top.get('stitle',''),
            'diamond_best_pident':top.get('pident',''),
            'diamond_best_query_coverage_hsp':top.get('qcovhsp',''),
            'diamond_best_subject_coverage_hsp':top.get('scovhsp',''),
            'diamond_best_evalue':top.get('evalue',''),
            'diamond_best_bitscore':top.get('bitscore',''),
            'interpro_has_any_match':s['has_any_match'],
            'interpro_has_mapped_entry':s['has_interpro_entry'],
            'interpro_match_objects':s['match_objects'],
            'interpro_integrated_entry_count':s['distinct_interpro_entries'],
            'interpro_integrated_accessions':s['interpro_accessions'],
            'interpro_signature_libraries':s['signature_libraries'],
            'interpro_has_go_annotation':s['has_go_annotation'],
            'annotation_coverage_category':label,
        })
        complete.append({
            'candidate_metadata':c,
            'diamond':{
                'query_summary':d,
                'reported_best_hit':top if top else None,
                'all_reported_alignments': grouped.get(k, []),
            },
            'interproscan':{
                'protein_summary':s,
                'original_full_result':ipidx[k],
            },
            'annotation_coverage_category':label,
        })
    need(len(complete) == len(summaries) == 552, 'Integrated count invalid')
    need(counters == Counter({'diamond_and_interpro_entry':343,'diamond_only':3,'interpro_entry_only':164,'neither_diamond_nor_interpro_entry':42}), 'Evidence coverage differs from independently audited records')
    no_ip = sum(int(r['interpro_has_any_match']) == 0 for r in summaries)
    need(no_ip == 9, 'InterProScan no-match count differs')
    need(sum(int(r['diamond_has_hit']) for r in summaries) == 346, 'DIAMOND hit count differs')
    need(sum(int(r['interpro_has_mapped_entry']) for r in summaries) == 507, 'InterProScan mapped count differs')
    print('HYPOSEQ 552 | INTEGRATION SOURCE AUDIT: PASS',flush=True)
    print('Candidates: 552; DIAMOND hits: 346; InterPro-mapped: 507; InterPro no-match: 9',flush=True)
    print('Coverage counts:', dict(counters),flush=True)
    if a.verify_only:
        print('VERIFY_ONLY: source data unchanged; no output written')
        return

    dest = root / BASE_REL / OUT_REL
    partial = root / BASE_REL / (OUT_REL.name + '.partial')
    need(not dest.exists(), 'Integration results already exist; inspect instead of overwriting: '+str(dest))
    need(not partial.exists(), 'Partial integration exists; inspect before rerunning: '+str(partial))
    need(not a.review_zip.exists(), 'Review ZIP already exists; inspect instead of overwriting: '+str(a.review_zip))
    partial.mkdir(parents=True)
    json_name = 'HYPOSEQ_552_INTEGRATED_EVIDENCE_v1.json'
    summary_name = 'HYPOSEQ_552_INTEGRATED_EVIDENCE_SUMMARY_v1.csv'
    cover_name = 'HYPOSEQ_552_EVIDENCE_COVERAGE_v1.json'
    audit_name = 'HYPOSEQ_552_INTEGRATION_AUDIT_v1.json'
    with (partial/json_name).open('w', encoding='utf-8') as f:
        json.dump({
            'project':'HypoSeq',
            'description':'Exact protein-key join of previously verified data; descriptive only; no AMR prioritization',
            'candidate_definition':candidate_manifest['candidate_definition'],
            'diamond_database':'reviewed UniProtKB/Swiss-Prot, release 2026_04',
            'interproscan_version':'6.0.3.0','interpro_version':'111.0',
            'source_sha256':sources,
            'proteins':complete,
        }, f, indent=1, ensure_ascii=False)
        f.write('\n')
    with (partial/summary_name).open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader(); w.writerows(summaries)
    with (partial/cover_name).open('w',encoding='utf-8') as f:
        json.dump({'total_candidates':552,'diamond_hit_candidates':346,
                   'interpro_mapped_candidates':507,'interpro_no_match_candidates':9,
                   'coverage_categories':dict(sorted(counters.items())),
                   'interpretation':'Annotation coverage only. Neither predicted AMR activity nor resistance phenotype is established. No biological ranking is performed.',
                   },f,indent=2); f.write('\n')
    audit = {
        'status':'PASS',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'query_count':552,
        'source_sha256':sources,
        'original_sequences_compared':552,
        'complete_protein_key_join':True,
        'diamond_rows_accounted_for':sum(len(x) for x in grouped.values()),
        'interproscan_protein_results_accounted_for':552,
        'interproscan_original_result_objects_preserved':True,
        'diamond_original_reported_alignment_rows_preserved':True,
        'protein_level_summary_count':len(summaries),
        'coverage_categories':dict(sorted(counters.items())),
        'biological_interpretation_performed':False,
        'priority_thresholds_applied':False,
        'source_directories_modified':False,
    }
    with (partial/audit_name).open('w',encoding='utf-8') as f:
        json.dump(audit,f,indent=2);f.write('\n')
    hashes = {p.name:sha_file(p) for p in partial.iterdir() if p.is_file()}
    with (partial/'OUTPUT_HASHES.json').open('w',encoding='utf-8') as f:
        json.dump(hashes,f,indent=2);f.write('\n')
    # Byte-for-byte audit of newly written integrated JSON and summaries before finalize.
    test=read_json(partial/json_name)
    need(len(test['proteins'])==552, 'Written JSON count invalid')
    need([r['candidate_metadata']['protein_key'] for r in test['proteins']] == list(cidx), 'Written JSON order differs')
    need(sum(len(r['diamond']['all_reported_alignments']) for r in test['proteins']) == sum(len(x) for x in grouped.values()), 'Written DIAMOND alignments changed')
    need(all(test['proteins'][i]['interproscan']['original_full_result'] == ipidx[candidates[i]['protein_key']] for i in range(552)), 'Written InterProScan objects changed')
    need(len(read_csv(partial/summary_name)) == 552, 'Written summary count invalid')
    for filename, digest in hashes.items():
        need(sha_file(partial/filename) == digest, 'Output checksum mismatch: '+filename)
    partial.rename(dest)
    # Download package is produced from the VERIFIED files saved on Drive, not scratch copies.
    with zipfile.ZipFile(a.review_zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(dest.iterdir()):
            if p.is_file(): z.write(p, arcname=p.name)
    print('HYPOSEQ 552 | INTEGRATED EVIDENCE SAVED: PASS')
    print('Results:', dest)
    print('Review ZIP:',a.review_zip)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('STOP: '+str(exc),file=sys.stderr)
        raise
