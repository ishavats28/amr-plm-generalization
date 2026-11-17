#!/usr/bin/env python3
"""Prepare auditable FASTA uploads for the frozen HypoSeq 552-candidate InterProScan web analysis.

Reads only the frozen export ZIP; writes only a new 02_interproscan/inputs directory.
No network access, annotation, inference, or modifications to source datasets.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

EXPECTED_ZIP_SHA256 = "796fa564a0c4574f7bd251e4eab5461b539635c8cf2306cb77808f533dc0577f"
EXPECTED_MEMBERS = {"M3_FROZEN_552_CANDIDATES.csv", "M3_FROZEN_552_CANDIDATES.faa", "EXPORT_MANIFEST.json"}
EXPECTED_COUNTS = [100, 100, 100, 100, 100, 52]
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
OUTPUT_NAME = "HypoSeq_552_InterProScan_Batches_v1.zip"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("STOP: " + message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fasta(data: bytes) -> list[tuple[str, str]]:
    text = data.decode("ascii")
    results = []
    identifier = None
    seq_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                results.append((identifier, "".join(seq_parts)))
            header = line[1:]
            require(header != "" and len(header.split()) == 1, "FASTA header must contain only the frozen protein ID")
            identifier = header
            seq_parts = []
        else:
            require(identifier is not None, "Sequence encountered before first FASTA header")
            seq_parts.append(line)
    if identifier is not None:
        results.append((identifier, "".join(seq_parts)))
    require(len(results) == 552, f"Frozen FASTA contains {len(results)} records instead of 552")
    require(len(set(key for key, _ in results)) == 552, "Duplicate FASTA IDs")
    for key, seq in results:
        require(bool(seq) and set(seq).issubset(CANONICAL_AA), f"Invalid amino acids in {key}")
    return results


def serialize_csv(rows: list[dict[str, str]], columns: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def prepare(source_zip: Path) -> dict[str, bytes]:
    require(source_zip.is_file(), f"Frozen source ZIP missing: {source_zip}")
    require(sha_file(source_zip) == EXPECTED_ZIP_SHA256, "Frozen candidate ZIP SHA-256 mismatch")
    with ZipFile(source_zip) as z:
        require(set(z.namelist()) == EXPECTED_MEMBERS and z.testzip() is None, "Unexpected or corrupted candidate ZIP")
        csv_content = z.read("M3_FROZEN_552_CANDIDATES.csv")
        fasta_content = z.read("M3_FROZEN_552_CANDIDATES.faa")
        provenance = json.loads(z.read("EXPORT_MANIFEST.json"))

    require(provenance.get("candidate_count") == 552, "Candidate count mismatch in export manifest")
    require(provenance.get("candidate_csv_sha256") == digest(csv_content), "CSV checksum mismatch")
    require(provenance.get("candidate_fasta_sha256") == digest(fasta_content), "FASTA checksum mismatch")
    require("M3_frozen_flag=1" in provenance.get("candidate_definition", ""), "Unexpected candidate definition")
    records = parse_fasta(fasta_content)

    reader = csv.DictReader(io.StringIO(csv_content.decode("utf-8")))
    require(reader.fieldnames is not None, "No CSV header")
    metadata = list(reader)
    require(len(metadata) == 552, "Unexpected CSV row count")
    require(len({row["protein_key"] for row in metadata}) == 552, "Duplicate metadata IDs")
    by_id = {row["protein_key"]: row for row in metadata}
    require(set(by_id) == {key for key, _ in records}, "FASTA/CSV protein identifiers mismatch")
    panel_counts = {}
    for key, sequence in records:
        row = by_id[key]
        panel_counts[row["panel_id"]] = panel_counts.get(row["panel_id"], 0) + 1
        require(digest(sequence.encode("ascii")) == row["sequence_sha256"], f"Sequence checksum mismatch: {key}")
        require(len(sequence) == int(row["length_aa"]), f"Sequence length mismatch: {key}")
        require(row["M3_frozen_flag"].strip().lower() in {"1", "true"}, f"Not M3-positive: {key}")
        require(row["amrf_any_hit"].strip().lower() == "false", f"AMRFinderPlus hit: {key}")
        require(row["exact_overlap_any"].strip().lower() == "false", f"Classification overlap: {key}")
        require("hypothetical protein" in row["source_header"].lower(), f"Unexpected source annotation: {key}")
    require(panel_counts == provenance.get("records_by_proteome"), "Per-proteome counts mismatch")

    result: dict[str, bytes] = {}
    mapping = []
    descriptions = []
    for index, count in enumerate(EXPECTED_COUNTS, 1):
        start = sum(EXPECTED_COUNTS[:index - 1])
        group = records[start:start + count]
        assert len(group) == count
        filename = f"HypoSeq_InterProScan_Batch_{index:02d}_of_06.fasta"
        content = "".join(f">{key}\n{seq}\n" for key, seq in group).encode("ascii")
        result[f"batches/{filename}"] = content
        descriptions.append({
            "batch_number": index,
            "filename": filename,
            "sequence_count": count,
            "first_protein_key": group[0][0],
            "last_protein_key": group[-1][0],
            "sha256": digest(content),
        })
        for position, (key, seq) in enumerate(group, 1):
            mapping.append({
                "protein_key": key,
                "panel_id": by_id[key]["panel_id"],
                "batch_number": str(index),
                "position_in_batch": str(position),
                "length_aa": str(len(seq)),
                "sequence_sha256": digest(seq.encode("ascii")),
                "batch_filename": filename,
            })
    require(len(mapping) == 552, "Batch map incomplete")
    map_bytes = serialize_csv(mapping, ["protein_key", "panel_id", "batch_number", "position_in_batch", "length_aa", "sequence_sha256", "batch_filename"])
    result["INTERPROSCAN_BATCH_MAP.csv"] = map_bytes
    manifest = {
        "project": "HypoSeq",
        "analysis": "552 frozen M3-positive hypothetical protein candidates; InterProScan website submissions",
        "source_candidate_zip_sha256": EXPECTED_ZIP_SHA256,
        "source_candidate_fasta_sha256": digest(fasta_content),
        "source_candidate_csv_sha256": digest(csv_content),
        "candidate_count": 552,
        "batch_count": 6,
        "batch_sizes": EXPECTED_COUNTS,
        "batching_rule": "Original frozen FASTA order, consecutive groups of up to 100; IDs and sequences unchanged",
        "reference": "https://www.ebi.ac.uk/interpro/",
        "result_analysis": "InterProScan web sequence search; NOT precomputed InterPro Matches API lookup",
        "batch_map_sha256": digest(map_bytes),
        "batches": descriptions,
        "notes": "Record InterProScan/InterPro and member-database versions from each actual completed job. Preserve complete JSON and TSV downloads, including zero-match sequences.",
    }
    result["INTERPROSCAN_BATCH_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    result["WEB_SUBMISSION_INSTRUCTIONS.txt"] = (
        "HypoSeq | frozen 552 InterProScan website submissions\n"
        "Website: https://www.ebi.ac.uk/interpro/\n"
        "Select Search by sequence; protein/amino-acid input. Upload each batches/*.fasta, one job per file.\n"
        "Each batch: at most 100 proteins. Keep standard analysis settings consistent across all six jobs.\n"
        "Record job ID, submission/completion dates, InterProScan version, member databases/versions, and any settings.\n"
        "Download full batch results, preferably JSON and TSV (also GFF/XML when offered), promptly.\n"
        "Save unmodified downloads separately for each batch; preserve jobs with no hits.\n"
    ).encode("utf-8")
    # Independent reparse of all newly generated FASTAs checks the exact 552-sequence coverage.
    reconstructed = []
    for desc in descriptions:
        reconstructed.extend(parse_batch(result["batches/" + desc["filename"]]))
    require(reconstructed == records, "Reassembled batch sequences differ from frozen candidate input")
    return result


def parse_batch(data: bytes) -> list[tuple[str, str]]:
    lines = data.decode("ascii").splitlines()
    pairs = []
    for n in range(0, len(lines), 2):
        require(n + 1 < len(lines) and lines[n].startswith(">"), "Invalid generated batch FASTA")
        pairs.append((lines[n][1:], lines[n + 1]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Original HypoSeq research root")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_dir(), f"Research root missing: {root}")
    source = root / "evaluation_v2/eskape_552_characterization_v1/00_provenance/HypoSeq_M3_Frozen_552_Hypothetical_Candidates.zip"
    target = root / "evaluation_v2/eskape_552_characterization_v1/02_interproscan/inputs"
    payload = prepare(source)
    payload_paths = {target / name: content for name, content in payload.items()}
    zip_target = target / OUTPUT_NAME
    # Guard against a partial, contaminated, or changed prior run.
    if target.exists():
        existing_files = {f for f in target.rglob("*") if f.is_file()}
        expected_files = set(payload_paths) | {zip_target}
        require(existing_files == expected_files, "Inputs directory exists with unexpected/missing files; inspect manually")
        for p, data in payload_paths.items():
            require(p.read_bytes() == data, f"Existing batch artifact mismatch: {p}")
        with ZipFile(zip_target) as z:
            require(set(z.namelist()) == set(payload), "Existing batch bundle entries mismatch")
            require(z.testzip() is None, "Existing batch bundle corrupted")
            for name, data in payload.items():
                require(z.read(name) == data, f"Existing batch bundle differs: {name}")
        print("HYPOSEQ | INTERPROSCAN BATCHES: VERIFIED_EXISTING", flush=True)
    else:
        # Validate in a scratch staging directory before writing anything to Drive.
        with TemporaryDirectory(prefix="hyposeq_interproscan_batches_") as tmp:
            stage = Path(tmp)
            for name, data in payload.items():
                p = stage / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            with ZipFile(stage / OUTPUT_NAME, "x", compression=ZIP_DEFLATED) as z:
                for name in sorted(payload):
                    z.write(stage / name, arcname=name)
            with ZipFile(stage / OUTPUT_NAME) as z:
                require(z.testzip() is None, "Staged batch bundle corrupted")
                require(set(z.namelist()) == set(payload), "Staged bundle incomplete")
            # Stage is fully validated; move files into new project directory, no overwrite.
            require(not target.exists(), "Input directory was created concurrently")
            target.mkdir(parents=True, exist_ok=False)
            for name in sorted(payload):
                p = target / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(payload[name])
                require(sha_file(p) == digest(payload[name]), f"Drive write mismatch: {name}")
            zip_target.write_bytes((stage / OUTPUT_NAME).read_bytes())
            with ZipFile(zip_target) as z:
                require(z.testzip() is None and set(z.namelist()) == set(payload), "Drive bundle verification failed")
        print("HYPOSEQ | INTERPROSCAN BATCHES: CREATED_AND_VERIFIED", flush=True)
    print("Batch counts: 100, 100, 100, 100, 100, 52", flush=True)
    print("Original 552 sequences preserved exactly", flush=True)
    print("Drive bundle:", zip_target, flush=True)
    print("Drive bundle SHA-256:", sha_file(zip_target), flush=True)


if __name__ == "__main__":
    main()
