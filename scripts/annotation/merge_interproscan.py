#!/usr/bin/env python3
"""HypoSeq: independently validate and merge six web InterProScan JSON/TSV batches.

This is a read-only source processing script. It validates the frozen 552 cohort,
original six submitted FASTAs, and all six JSON/TSV pairs; outputs ONLY into a new,
previously absent output directory. No annotation or biological interpretation is
added, and JSON result objects are preserved without filtering or modification.

Example:
  python scripts/annotation/merge_interproscan.py \
    --candidate-zip HypoSeq_M3_Frozen_552_Hypothetical_Candidates.zip \
    --prepared-zip HypoSeq_552_InterProScan_Preparation_v1.zip \
    --results-zip 'InterPro Results.zip' \
    --out-dir outputs/interproscan_merge
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

CANDIDATES_SHA256 = '796fa564a0c4574f7bd251e4eab5461b539635c8cf2306cb77808f533dc0577f'
EXPECTED_COUNTS = [100, 100, 100, 100, 100, 52]
EXPECTED_RAW_NAMES = {f'batch_{i:02}.{ext}' for i in range(1, 7) for ext in ('json', 'tsv')}
OUTPUT_PREFIX = 'HypoSeq_552_InterProScan'


def require(condition, message):
    if not condition:
        raise RuntimeError('STOP: ' + message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fasta_dict(raw):
    seqs = {}
    k = None
    for line in raw.decode('ascii').splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            k = line[1:].split()[0]
            require(bool(k) and k not in seqs, 'Malformed or duplicate FASTA identifier')
            seqs[k] = ''
        else:
            require(k is not None, 'FASTA starts without sequence identifier')
            seqs[k] += line
    return seqs


def as_csv_bytes(rows, headers):
    io_ = io.StringIO(newline='')
    w = csv.DictWriter(io_, fieldnames=headers, lineterminator='\n')
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return io_.getvalue().encode('utf-8')


def run(candidate_zip, prepared_zip, results_zip, out_dir):
    out_dir = Path(out_dir).resolve()
    require(not out_dir.exists(), f'Output directory exists (not overwriting): {out_dir}')
    require(sha_file(candidate_zip) == CANDIDATES_SHA256,
            'Frozen candidate ZIP SHA-256 differs from the reconciled cohort')

    with ZipFile(candidate_zip) as cz:
        require(cz.testzip() is None, 'Candidate ZIP integrity failure')
        manifest = json.loads(cz.read('EXPORT_MANIFEST.json'))
        canonical = cz.read('M3_FROZEN_552_CANDIDATES.faa')
        original = fasta_dict(canonical)
        require(len(original) == 552 and manifest['candidate_count'] == 552,
                'Frozen candidate count mismatch')
        require(sha256(canonical) == manifest['candidate_fasta_sha256'],
                'Frozen FASTA checksum mismatch against export manifest')
        metadata = cz.read('M3_FROZEN_552_CANDIDATES.csv')
        require(sha256(metadata) == manifest['candidate_csv_sha256'],
                'Frozen metadata checksum mismatch against export manifest')
        mrows = list(csv.DictReader(io.StringIO(metadata.decode('utf-8'))))
        require(len(mrows) == 552 and {r['protein_key'] for r in mrows} == set(original),
                'Frozen metadata protein IDs differ from FASTA')
        require(all(r['M3_frozen_flag'] in ('1', '1.0') and
                    str(r['amrf_any_hit']).lower() == 'false' and
                    str(r['exact_overlap_any']).lower() == 'false' and
                    'hypothetical protein' in r['source_header'].lower()
                    for r in mrows), 'Unexpected candidate metadata selection flags')

    with ZipFile(prepared_zip) as pz:
        require(pz.testzip() is None, 'Preparation ZIP integrity failure')
        batch_manifest = json.loads(pz.read('INTERPROSCAN_BATCH_MANIFEST.json'))
        require(batch_manifest.get('candidate_count') == 552 and
                batch_manifest.get('batch_sizes') == EXPECTED_COUNTS,
                'Batch manifest differs from frozen six-batch plan')
        prepared_ids = set()
        prepared_batches = {}
        for i, rec in enumerate(batch_manifest['batches'], 1):
            expected_name = f'HypoSeq_InterProScan_Batch_{i:02}_of_06.fasta'
            require(rec['batch_number'] == i and rec['filename'] == expected_name,
                    f'Unexpected preparation batch {i}')
            raw = pz.read('batches/' + expected_name)
            require(sha256(raw) == rec['sha256'], f'Batch FASTA hash mismatch: {i}')
            subset = fasta_dict(raw)
            require(len(subset) == EXPECTED_COUNTS[i-1] and
                    rec['sequence_count'] == len(subset), f'Batch size mismatch {i}')
            require(set(subset).isdisjoint(prepared_ids), f'Duplicate protein across batches {i}')
            prepared_ids.update(subset)
            require(all(original.get(k) == s for k, s in subset.items()),
                    f'Batch {i} has sequence outside original frozen cohort')
            prepared_batches[i] = subset
        require(prepared_ids == set(original), 'Batches do not cover exactly 552 proteins')
        batch_map = pz.read('INTERPROSCAN_BATCH_MAP.csv')
        require(sha256(batch_map) == batch_manifest['batch_map_sha256'],
                'Batch map checksum mismatch')

    batch_reports = []
    records = {}
    protein_batches = {}
    version = None
    ipversion = None
    total_locations = 0
    total_matches = 0
    signature_source_counts = Counter()
    match_library_versions = defaultdict(set)
    tsv_dates = set()
    result_json_hashes = {}
    result_tsv_hashes = {}

    with ZipFile(results_zip) as rz:
        require(rz.testzip() is None, 'Raw InterProScan results ZIP is corrupt')
        require(set(rz.namelist()) == EXPECTED_RAW_NAMES,
                'Result ZIP must contain exactly batch_01..06.json and .tsv (no extra files)')
        for i in range(1, 7):
            name_j, name_t = f'batch_{i:02}.json', f'batch_{i:02}.tsv'
            data_j, data_t = rz.read(name_j), rz.read(name_t)
            result_json_hashes[name_j] = sha256(data_j)
            result_tsv_hashes[name_t] = sha256(data_t)
            doc = json.loads(data_j)
            require(isinstance(doc, dict) and
                    set(doc) == {'interproscan-version', 'interpro-version', 'results'},
                    f'Unexpected JSON wrapper in batch {i}')
            if version is None:
                version, ipversion = doc['interproscan-version'], doc['interpro-version']
            require(doc['interproscan-version'] == version and
                    doc['interpro-version'] == ipversion,
                    f'InterProScan/InterPro release mismatch in batch {i}')
            results = doc['results']
            require(isinstance(results, list) and len(results) == EXPECTED_COUNTS[i-1],
                    f'Batch {i} JSON must include all {EXPECTED_COUNTS[i-1]} queries')
            expected_batch = prepared_batches[i]
            seen = set()
            json_loc_keys = Counter()
            nonempty = 0
            interpro_mapped = 0
            bm = 0
            locations_count = 0
            for r in results:
                require(isinstance(r, dict) and
                        set(r) >= {'xref', 'sequence', 'md5', 'matches'},
                        f'Unexpected protein record structure in batch {i}')
                xr = r['xref']
                require(isinstance(xr, list) and len(xr) == 1 and
                        xr[0].get('id') == xr[0].get('name') and
                        isinstance(xr[0].get('id'), str),
                        f'Ambiguous protein identifier in batch {i}')
                pid = xr[0]['id']
                require(pid in expected_batch and pid not in seen and pid not in records,
                        f'Unexpected/duplicate protein in batch {i}: {pid}')
                seen.add(pid)
                require(r['sequence'] == original[pid], f'Sequence changed: {pid}')
                require(r['md5'].upper() == hashlib.md5(original[pid].encode('ascii')).hexdigest().upper(),
                        f'InterProScan MD5 mismatch: {pid}')
                require(isinstance(r['matches'], list), f'Invalid matches array: {pid}')
                nonempty += bool(r['matches'])
                mapped_here = False
                for m in r['matches']:
                    require(isinstance(m, dict) and isinstance(m.get('signature'), dict),
                            f'Invalid signature structure: {pid}')
                    s = m['signature']
                    release = s.get('signatureLibraryRelease') or {}
                    lib = release.get('library') or m.get('source')
                    require(isinstance(lib, str) and bool(lib), f'Missing source library: {pid}')
                    signature_source_counts[m.get('source') or ''] += 1
                    if release.get('version'):
                        match_library_versions[lib].add(str(release['version']))
                    entry = s.get('entry')
                    if isinstance(entry, dict) and entry.get('accession'):
                        require(str(entry['accession']).startswith('IPR'),
                                f'Unexpected integrated InterPro accession in {pid}')
                        mapped_here = True
                    require(isinstance(m.get('locations'), list), f'No locations list: {pid}')
                    bm += 1
                    for loc in m['locations']:
                        a, b = loc.get('start'), loc.get('end')
                        require(type(a) is int and type(b) is int and 1 <= a <= b <= len(original[pid]),
                                f'Invalid location coordinates for {pid}')
                        json_loc_keys[(pid, lib, s.get('accession'), a, b)] += 1
                        locations_count += 1
                interpro_mapped += mapped_here
                records[pid] = r
                protein_batches[pid] = i
            require(seen == set(expected_batch), f'Batch {i} has missing/extra protein IDs')
            tsv_loc_keys = Counter()
            tsv_rows = 0
            for line in data_t.decode('utf-8-sig').splitlines():
                if not line.strip():
                    continue
                cols = line.split('\t')
                require(len(cols) == 15, f'Unexpected TSV column count in batch {i}')
                pid, md5, length = cols[:3]
                require(pid in expected_batch and
                        md5.upper() == hashlib.md5(original[pid].encode('ascii')).hexdigest().upper() and
                        int(length) == len(original[pid]),
                        f'TSV query or checksum discrepancy batch {i}: {pid}')
                require(cols[6].isdigit() and cols[7].isdigit(),
                        f'Unexpected TSV start/end coordinates batch {i}: {pid}')
                tsv_loc_keys[(pid, cols[3], cols[4], int(cols[6]), int(cols[7]))] += 1
                tsv_dates.add(cols[10])
                tsv_rows += 1
            require(tsv_loc_keys == json_loc_keys,
                    f'TSV and JSON annotation locations disagree in batch {i}')
            require(tsv_rows == locations_count,
                    f'TSV count differs from JSON location count in batch {i}')
            total_locations += locations_count
            total_matches += bm
            batch_reports.append({
                'batch_number': i,
                'protein_records': len(results),
                'proteins_with_any_match': nonempty,
                'proteins_with_interpro_entry': interpro_mapped,
                'proteins_with_no_match': len(results)-nonempty,
                'match_objects': bm,
                'annotation_locations': locations_count,
                'tsv_rows': tsv_rows,
                'json_sha256': result_json_hashes[name_j],
                'tsv_sha256': result_tsv_hashes[name_t],
            })
    require(set(records) == set(original) and len(records) == 552,
            'Merged results do not cover exactly the 552 frozen candidates')
    require(version == '6.0.3.0' and ipversion == '111.0',
            'InterProScan/InterPro versions differ from the audited raw batch set')

    summary_rows = []
    no_match_rows = []
    matched_count = 0
    mapped_count = 0
    go_count = 0
    for pid, seq in original.items():
        r = records[pid]
        match_count = len(r['matches'])
        libs, interpro_ids, gos, paths, member_signatures = set(), set(), set(), set(), set()
        for m in r['matches']:
            s = m['signature']; rel = s.get('signatureLibraryRelease') or {}
            libs.add(rel.get('library') or m.get('source'))
            member_signatures.add((rel.get('library') or m.get('source'), s.get('accession')))
            entry = s.get('entry')
            if isinstance(entry, dict) and entry.get('accession'):
                interpro_ids.add(entry['accession'])
                gos.update(str(x.get('id')) for x in entry.get('goXRefs', [])
                           if isinstance(x, dict) and x.get('id'))
                paths.update(str(x.get('id')) for x in entry.get('pathwayXRefs', [])
                             if isinstance(x, dict) and x.get('id'))
        matched_count += bool(match_count)
        mapped_count += bool(interpro_ids)
        go_count += bool(gos)
        summary_rows.append({
            'protein_key':pid,
            'batch':protein_batches[pid],
            'length_aa':len(seq),
            'match_objects':match_count,
            'has_any_match':int(bool(match_count)),
            'has_interpro_entry':int(bool(interpro_ids)),
            'distinct_interpro_entries':len(interpro_ids),
            'distinct_member_signatures':len(member_signatures),
            'has_go_annotation':int(bool(gos)),
            'interpro_accessions':';'.join(sorted(interpro_ids)),
            'signature_libraries':';'.join(sorted(x for x in libs if x)),
        })
        if not match_count:
            no_match_rows.append({'protein_key':pid, 'batch':protein_batches[pid],
                                  'length_aa':len(seq)})
    require(matched_count == 543 and mapped_count == 507 and len(no_match_rows) == 9,
            'Unexpected complete-run match counts; inspect batch data before merge')

    merged = {
        'interproscan-version': version,
        'interpro-version': ipversion,
        # Original frozen FASTA order, not the order returned by InterProScan.
        'results': [records[pid] for pid in original],
    }
    combined_data = json.dumps(merged, ensure_ascii=False,
                               separators=(',', ':')).encode('utf-8')
    require(len(json.loads(combined_data)['results']) == 552,
            'Combined JSON round-trip count mismatch')

    sum_headers = list(summary_rows[0])
    summary_data = as_csv_bytes(summary_rows, sum_headers)
    no_match_data = as_csv_bytes(no_match_rows, ['protein_key', 'batch', 'length_aa'])
    total_per_batch = [r['annotation_locations'] for r in batch_reports]
    overview = {
        'project': 'HypoSeq',
        'analysis': 'InterProScan 6 web submission of frozen M3-positive 552 hypothetical proteins',
        'source_definition': 'Original 552-candidate, six-batch InterProScan website results',
        'interproscan_version': version,
        'interpro_version': ipversion,
        'candidate_count': len(original),
        'batch_counts': EXPECTED_COUNTS,
        'proteins_with_any_match': matched_count,
        'proteins_with_interpro_entry': mapped_count,
        'proteins_without_matches': len(no_match_rows),
        'proteins_with_go_annotation': go_count,
        'match_objects': total_matches,
        'annotation_locations': total_locations,
        'tsv_rows_by_batch': total_per_batch,
        'signature_source_counts': dict(sorted(signature_source_counts.items())),
        'member_database_versions': {k: sorted(v) for k, v in sorted(match_library_versions.items())},
        'tsv_reported_dates': sorted(tsv_dates),
        'batch_audit': batch_reports,
        'notes': [
            'All source JSON result objects were preserved without filtering or changes.',
            'Combined JSON record order follows original frozen FASTA, independent of web-returned result order.',
            'InterPro entry mapping is distinct from an arbitrary signature or sequence-feature match.',
            'Sequence-based functional annotation does not confirm antimicrobial resistance phenotypes.',
            'Raw six JSON and six TSV files should be archived unchanged separately.',
            'Job IDs were not included in the supplied file archive; record separately if available.',
        ],
    }
    overview_data = (json.dumps(overview, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    candidate_sha = sha_file(candidate_zip)
    prepared_sha = sha_file(prepared_zip)
    raw_sha = sha_file(results_zip)
    audit_text = f'''# HypoSeq | Independent InterProScan merge audit\n\n**Result: PASS**\n\n- Frozen 552-candidate ZIP SHA-256: `{candidate_sha}`\n- Original preparation archive SHA-256: `{prepared_sha}`\n- Uploaded raw InterPro results archive SHA-256: `{raw_sha}`\n- InterProScan version: `{version}`; InterPro version: `{ipversion}`\n- Batches: 100, 100, 100, 100, 100, 52; all returned protein IDs unique and batch-correct\n- All 552 sequences exactly match the frozen input; 552/552 MD5 checks passed\n- JSON/TSV location correspondence: {total_locations:,}/{total_locations:,}; no discrepancies\n- Proteins with any match: {matched_count}; with mapped InterPro entry: {mapped_count}; without any match: {len(no_match_rows)}\n- Protein records in output: 552, in original frozen FASTA order\n- No annotation filtering, evidence ranking, or AMR functional assignment performed\n\n## Data interpretation\n\nSignature presence is not equivalent to a mapped InterPro entry or validated biochemical function. Hits from topology/feature predictors and InterPro-N should be interpreted in their source-specific context.\n\n## Reproducibility\n\nKeep raw input ZIP (six JSON and six TSV) unchanged. The merged JSON is a lossless protein-record-level reorganization with the same original root version fields, while the companion manifest records the source hashes and all verification counts.\n\nSubmitter's InterPro job identifiers were not included in the uploaded ZIP and should be saved separately if available.\n'''
    files = {
        f'{OUTPUT_PREFIX}_Combined_v1.json': combined_data,
        f'{OUTPUT_PREFIX}_Protein_Summary_v1.csv': summary_data,
        f'{OUTPUT_PREFIX}_No_Match_Proteins_v1.csv': no_match_data,
        f'{OUTPUT_PREFIX}_Summary_Stats_v1.json': overview_data,
        f'{OUTPUT_PREFIX}_Merge_Audit_v1.md': audit_text.encode('utf-8'),
    }
    manifest_out = {
        'project':'HypoSeq',
        'status':'PASS',
        'interproscan_version':version,
        'interpro_version':ipversion,
        'source_files':{
            'frozen_candidate_zip_sha256':candidate_sha,
            'prepared_batch_zip_sha256':prepared_sha,
            'uploaded_interpro_results_zip_sha256':raw_sha,
            'raw_json_sha256_by_batch':result_json_hashes,
            'raw_tsv_sha256_by_batch':result_tsv_hashes,
        },
        'output_files_sha256':{name:sha256(value) for name,value in sorted(files.items())},
        'validation':{
            'source_fasta_count':552,
            'merged_json_count':len(records),
            'matching_sequence_md5':552,
            'missing_protein_count':0,
            'duplicate_protein_count':0,
            'json_tsv_location_agreement_count':total_locations,
            'proteins_with_any_match':matched_count,
            'proteins_with_integrated_interpro_entry':mapped_count,
            'proteins_without_matches':len(no_match_rows),
        },
        'combined_file_preservation':'All original result dictionaries unfiltered; ordered by frozen FASTA',
        'raw_batches_preserved_in_source_archive':True,
        'interpro_job_ids_in_uploaded_data':False,
    }
    files[f'{OUTPUT_PREFIX}_Merge_Manifest_v1.json'] = (
        json.dumps(manifest_out, indent=2, ensure_ascii=False) + '\n').encode('utf-8')

    out_dir.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (out_dir/name).write_bytes(data)
    zip_name = out_dir/f'{OUTPUT_PREFIX}_Merged_Package_v1.zip'
    with ZipFile(zip_name, 'x', compression=ZIP_DEFLATED, compresslevel=7) as zo:
        for name, data in sorted(files.items()):
            zo.writestr(name, data)
    with ZipFile(zip_name) as zr:
        require(zr.testzip() is None and set(zr.namelist()) == set(files),
                'Merged package corrupt')
        require(all(zr.read(name) == data for name, data in files.items()),
                'Merged ZIP content mismatch')
    print('HYPOSEQ | INTERPROSCAN SIX-BATCH MERGE: PASS')
    print('Proteins:', len(records), 'InterProScan:', version, 'InterPro:', ipversion)
    print('Any matches:', matched_count, 'InterPro-mapped:', mapped_count,
          'No matches:', len(no_match_rows), 'Locations:', total_locations)
    print('Merged package:', zip_name)
    print('Merged JSON SHA-256:', sha256(combined_data))
    return zip_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-zip', type=Path, required=True)
    ap.add_argument('--prepared-zip', type=Path, required=True)
    ap.add_argument('--results-zip', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    run(a.candidate_zip, a.prepared_zip, a.results_zip, a.out_dir)

if __name__ == '__main__':
    main()
