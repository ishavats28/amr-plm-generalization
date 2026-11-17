"""Dataset integrity and provenance checks shared by frozen-encoder training workflows.

Only train and validation data are ever opened by modeling code.
Test and OOD belong to the separately certified frozen audit, not model training.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

REPORT_SHA = "f587a237fdf4a48b1beaa559cdd56e2461cf46b600b1038e3c5969eb29a6e8ac"
EXPECTED = {
    "train": (6076, 6076),  # negative, positive
    "validation": (750, 750),
}
SPLIT_HASHES = {
    "train": "92a15894ffb2e7cf2777b2607d8de817e2803b75ccdc36b3ec8bdd5863d7a65a",
    "validation": "12628789ecdeabff652d5e49d82f9c92f3a71671340f57bd5480297361f88b63",
}


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def ids_digest(frame):
    return sha_bytes(("\n".join(frame["protein_id"].tolist()) + "\n").encode("ascii"))


def verify_attestation(root: Path):
    root = Path(root)
    report_path = root / "metadata/v2_split_audit.json"
    manifest_path = root / "metadata/v2_split_manifest.json"
    auditor_path = root / "split_scripts/audit_v2_split.py"
    design_path = root / "MANUSCRIPT_NOTES_v2.md"
    require(report_path.is_file() and manifest_path.is_file(), "Certified split audit report or manifest missing")
    require(sha256(report_path) == REPORT_SHA, "independent audit report SHA256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(report.get("status") == "PASS", "independent audit status is not PASS")
    checks = report.get("checks")
    require(isinstance(checks, list) and len(checks) == 150 and all(
        isinstance(c, dict) and c.get("passed") is True for c in checks
    ), "independent audit checks not all PASS")
    require(auditor_path.is_file() and sha256(auditor_path) == report.get("audit_script_sha256"),
            "auditor code has changed since certification")
    require(sha256(manifest_path) == report.get("manifest_sha256_before_audit"),
            "frozen manifest changed since certification")
    require(design_path.is_file() and sha256(design_path) == report.get("design_note_sha256"),
            "frozen design note changed since certification")
    require(manifest.get("schema") == "amr_binary_study_v2_split_v1", "unexpected manifest schema")
    require(report.get("blast_replay", {}).get("qualifying_rows") == 2234332,
            "historical BLAST reproduction missing")
    require(report.get("blast_replay", {}).get("bidirectional_qualifying_cross_partition_rows") == 0,
            "bidirectional cross-partition BLAST audit not clean")
    require(report.get("resistance_category_audit", {}).get("matches_frozen_manifest") is True,
            "resistance-category inventory not reconciled")
    require(manifest.get("held_out_predictions_generated", {}).get("test") is False and
            manifest.get("held_out_predictions_generated", {}).get("ood") is False,
            "manifest held-out prediction state unexpected")
    return report, manifest


def load_development(root: Path):
    """Only read and validate the frozen train/validation files. Never read test/OOD."""
    root = Path(root)
    report, manifest = verify_attestation(root)
    result = {}
    seen_ids, component_to_split, old_cluster_to_split = set(), {}, {}
    for name, expected_counts in EXPECTED.items():
        rel = f"data/processed/{name}.tsv.gz"
        path = root / rel
        require(path.is_file(), f"missing {rel}")
        observed = sha256(path)
        require(observed == SPLIT_HASHES[name], f"{name} dataset SHA256 differs from certified hash")
        require(observed == manifest["output_hashes"][rel] == report["verified_output_hashes"][rel],
                f"manifest/report dataset SHA256 mismatch: {rel}")
        df = pd.read_csv(path, sep="\t", dtype={"protein_id": str, "sequence": str})
        required = {"protein_id", "sequence", "length", "label", "homology_component", "old_cdhit_cluster", "split"}
        require(required.issubset(df.columns), f"{name}: required columns missing")
        require(df["protein_id"].notna().all() and df["protein_id"].is_unique,
                f"{name}: null/duplicate IDs")
        require(df["sequence"].notna().all() and (df["sequence"].str.len() == df["length"]).all(),
                f"{name}: invalid sequence/length")
        require(df["sequence"].map(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY")).all(),
                f"{name}: noncanonical amino acid found; inspect before encoding")
        require((df["split"] == name).all(), f"{name}: split-column mismatch")
        require(df["label"].isin([0, 1]).all(), f"{name}: unexpected label")
        counts = df["label"].value_counts().to_dict()
        require((counts.get(0, 0), counts.get(1, 0)) == expected_counts,
                f"{name}: class counts mismatch {counts}")
        require(df["homology_component"].notna().all() and df["old_cdhit_cluster"].notna().all(),
                f"{name}: missing homology identifiers")
        ids = set(df["protein_id"])
        require(not seen_ids.intersection(ids), "duplicate ID between train/validation")
        seen_ids.update(ids)
        for col, mapping in (("homology_component", component_to_split),
                             ("old_cdhit_cluster", old_cluster_to_split)):
            for c in df[col].unique():
                require(c not in mapping or mapping[c] == name, f"train/validation {col} overlap: {c}")
                mapping[c] = name
        result[name] = df.sort_values("protein_id", kind="stable").reset_index(drop=True)
    return result


def atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.with_suffix(path.suffix + ".partial").exists(), f"stale partial: {path}")
    data = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_bytes(path, data.encode("utf-8"))


def _atomic_bytes(path: Path, data: bytes):
    path = Path(path)
    stage = path.with_name(path.name + ".partial")
    require(not stage.exists(), f"stale partial file {stage}")
    with stage.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(stage, path)


def require_no_partial(directory: Path):
    if directory.exists():
        flagged = [p for p in directory.rglob("*") if p.is_file() and p.name.endswith((".partial", ".tmp"))]
        require(not flagged, f"stale partial file(s) in {directory}: {flagged[:5]}")
