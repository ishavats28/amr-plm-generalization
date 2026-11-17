from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import tarfile
import tempfile

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "."
)

POS_PATH = (
    ROOT
    / "data"
    / "interim"
    / "positive_model_range.tsv.gz"
)

NEG_PATH = (
    ROOT
    / "data"
    / "interim"
    / "negative_model_range.tsv.gz"
)

EVIDENCE_PATH = (
    ROOT
    / "metadata"
    / "preprocessing_evidence.tar.gz"
)

ASSIGNMENT_PATH = (
    ROOT
    / "metadata"
    / "v2_split_assignment.tsv.gz"
)

GROUPS_PATH = (
    ROOT
    / "metadata"
    / "v2_biological_groups.tsv.gz"
)

MANIFEST_PATH = (
    ROOT
    / "metadata"
    / "v2_split_manifest.json"
)

DESIGN_NOTE_PATH = (
    ROOT
    / "MANUSCRIPT_NOTES_v2.md"
)

AMRF_CATALOG_PATH = (
    ROOT
    / "data"
    / "raw"
    / "amrfinderplus"
    / "ReferenceGeneCatalog.txt"
)

AMRF_HIERARCHY_PATH = (
    ROOT
    / "data"
    / "raw"
    / "amrfinderplus"
    / "ReferenceGeneHierarchy.txt"
)

CARD_FASTA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "card"
    / "protein_fasta_protein_homolog_model.fasta"
)

CARD_JSON_PATH = (
    ROOT
    / "data"
    / "raw"
    / "card"
    / "card.json"
)

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train.tsv.gz"
)

VALIDATION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "validation.tsv.gz"
)

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "test.tsv.gz"
)

OOD_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ood.tsv.gz"
)

AUDIT_REPORT_PATH = (
    ROOT
    / "metadata"
    / "v2_split_audit.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_TOTAL = 15794
EXPECTED_POS = 7897
EXPECTED_NEG = 7897

EXPECTED_COMPONENTS = 2688
EXPECTED_POS_ONLY_COMPONENTS = 339
EXPECTED_NEG_ONLY_COMPONENTS = 2339
EXPECTED_MIXED_COMPONENTS = 10

EXPECTED_AMRF_MAPPED_POSITIVES = 7568
EXPECTED_AMRF_ANCHORED_COMPONENTS = 185

EXPECTED_PRIMARY_ANCHORED_COMPONENTS = 281
EXPECTED_PRIMARY_ANCHORED_PROTEINS = 5257

EXPECTED_ZERO_AMRF_COMPONENTS = 98

EXPECTED_CARD_FAMILY_ANCHORED_COMPONENTS = 73
EXPECTED_CARD_MECHANISM_ANCHORED_COMPONENTS = 23
EXPECTED_CARD_UNANCHORED_COMPONENTS = 2

EXPECTED_OOD_BLOCKS = 24
EXPECTED_OOD_POS_COMPONENTS = 29
EXPECTED_OOD_POS_PROTEINS = 321

EXPECTED_ELIGIBLE_POS_COMPONENTS = 278
EXPECTED_ELIGIBLE_POS_PROTEINS = 1935

EXPECTED_AMRF_SEEDS = 45
EXPECTED_CARD_SEEDS = 2

# The historical preprocessing counted only a canonical directed BLAST
# alignment: shorter protein as query; for equal lengths, qseqid < sseqid.
# This count is NOT the number of all qualifying directed alignments.
EXPECTED_QUALIFYING_BLAST_ROWS = 2234332
EXPECTED_ALL_VS_ALL_BLAST_ROWS = 7890489
EXPECTED_BIDIRECTIONAL_QUALIFYING_ROWS = 4472696
EXPECTED_BIDIRECTIONAL_SELF_ROWS = 15794

PRIMARY_COMPONENT_CAP = 112
OOD_NEGATIVE_COMPONENT_CAP = 48

KS_NUMERATOR = 1
KS_DENOMINATOR = 10


# ============================================================
# UTILITIES
# ============================================================

CHECKS = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def clean(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if value == "":
        return None

    if value.lower() in {
        "nan",
        "none",
        "null",
        "na",
    }:
        return None

    return value


def check(
    name: str,
    condition: bool,
    details: str = "",
):
    condition = bool(condition)

    CHECKS.append({
        "name": name,
        "passed": condition,
        "details": str(details),
    })

    print(
        f"{name}: "
        f"{'PASS' if condition else 'FAIL'}"
    )

    if details:
        print(
            f"  {details}"
        )

    if not condition:
        raise AssertionError(
            f"{name} failed. {details}"
        )


def split_accessions(value):
    value = clean(value)

    if value is None:
        return set()

    tokens = set()

    for part in re.split(
        r"[;,\s|]+",
        value,
    ):
        part = part.strip(
            "[](){}<>"
        )

        if not part:
            continue

        tokens.add(part)

        if (
            ":"
            in part
            and not part.startswith(
                "ARO:"
            )
        ):
            tokens.add(
                part.rsplit(
                    ":",
                    1,
                )[-1]
            )

    for aro in re.findall(
        r"ARO:(\d+)",
        value,
    ):
        tokens.add(
            f"ARO:{aro}"
        )

    return tokens


def norm_acc(value):
    """
    AMRFinder/CARD protein-accession normalization used by
    the split builder: ignore terminal numeric version suffix.
    """
    value = clean(value)

    if value is None:
        return None

    return re.sub(
        r"\.\d+$",
        "",
        value,
    )


def normalize_aro(value):
    value = clean(value)

    if value is None:
        return None

    m = re.search(
        r"(\d+)",
        value,
    )

    if m is None:
        return None

    return m.group(1)


def exact_ks(a, b):
    a = np.sort(
        np.asarray(
            a,
            dtype=np.int64,
        )
    )

    b = np.sort(
        np.asarray(
            b,
            dtype=np.int64,
        )
    )

    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            "KS comparison received an empty sample."
        )

    values = np.union1d(
        a,
        b,
    )

    ca = np.searchsorted(
        a,
        values,
        side="right",
    ).astype(
        np.int64
    )

    cb = np.searchsorted(
        b,
        values,
        side="right",
    ).astype(
        np.int64
    )

    na = int(len(a))
    nb = int(len(b))

    deviations = np.abs(
        ca * nb
        - cb * na
    )

    max_dev = int(
        deviations.max()
    )

    denominator = (
        na * nb
    )

    value = (
        max_dev
        / denominator
    )

    return {
        "n_a": na,
        "n_b": nb,
        "cross_multiplied_deviation":
            max_dev,
        "denominator":
            denominator,
        "ks":
            float(value),
        "leq_0_10_exact":
            (
                max_dev
                * KS_DENOMINATOR
                <= denominator
                * KS_NUMERATOR
            ),
    }


def component_kind(
    positives,
    negatives,
):
    if positives > 0 and negatives == 0:
        return "positive_only"

    if positives == 0 and negatives > 0:
        return "negative_only"

    return "mixed"


# ============================================================
# PREFLIGHT
# ============================================================

print("=" * 80)
print("INDEPENDENT SPLIT AUDIT")
print("=" * 80)

required_files = [
    POS_PATH,
    NEG_PATH,
    EVIDENCE_PATH,
    ASSIGNMENT_PATH,
    GROUPS_PATH,
    MANIFEST_PATH,
    DESIGN_NOTE_PATH,
    AMRF_CATALOG_PATH,
    AMRF_HIERARCHY_PATH,
    CARD_FASTA_PATH,
    CARD_JSON_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    OOD_PATH,
]

for path in required_files:
    check(
        f"File exists: {path.relative_to(ROOT)}",
        path.is_file(),
    )

check(
    "Audit report does not already exist",
    not AUDIT_REPORT_PATH.exists(),
    (
        "Refusing to overwrite an existing "
        "independent audit report."
    ),
)


manifest = json.loads(
    MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)

manifest_hash = sha256_file(
    MANIFEST_PATH
)

audit_script_hash = sha256_file(
    Path(__file__)
)

check(
    "Manifest status is pre-audit",
    manifest.get("status")
    == "generated_not_independently_audited",
)

check(
    "Manifest independent_audit_passed is false",
    manifest.get(
        "independent_audit_passed"
    ) is False,
)

check(
    "Manifest says test predictions absent",
    manifest[
        "held_out_predictions_generated"
    ][
        "test"
    ] is False,
)

check(
    "Manifest says OOD predictions absent",
    manifest[
        "held_out_predictions_generated"
    ][
        "ood"
    ] is False,
)


# ============================================================
# HASH AUDIT
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("HASH AUDIT")
print("=" * 80)

input_path_map = {
    "ReferenceGeneCatalog.txt":
        AMRF_CATALOG_PATH,

    "ReferenceGeneHierarchy.txt":
        AMRF_HIERARCHY_PATH,

    "card.json":
        CARD_JSON_PATH,

    "negative_model_range.tsv.gz":
        NEG_PATH,

    "positive_model_range.tsv.gz":
        POS_PATH,

    "preprocessing_evidence.tar.gz":
        EVIDENCE_PATH,

    "protein_fasta_protein_homolog_model.fasta":
        CARD_FASTA_PATH,
}

for key, path in input_path_map.items():
    observed = sha256_file(
        path
    )

    expected = manifest[
        "input_hashes"
    ][key]

    check(
        f"Input hash: {key}",
        observed == expected,
        f"{observed}",
    )


check(
    "Design-note hash",
    sha256_file(
        DESIGN_NOTE_PATH
    )
    == manifest[
        "design_note"
    ][
        "sha256"
    ],
)

output_path_map = {
    "data/processed/train.tsv.gz":
        TRAIN_PATH,

    "data/processed/validation.tsv.gz":
        VALIDATION_PATH,

    "data/processed/test.tsv.gz":
        TEST_PATH,

    "data/processed/ood.tsv.gz":
        OOD_PATH,

    "metadata/v2_split_assignment.tsv.gz":
        ASSIGNMENT_PATH,

    "metadata/v2_biological_groups.tsv.gz":
        GROUPS_PATH,
}

observed_output_hashes = {}

for key, path in output_path_map.items():
    observed = sha256_file(
        path
    )

    observed_output_hashes[
        key
    ] = observed

    expected = manifest[
        "output_hashes"
    ][key]

    check(
        f"Output hash: {key}",
        observed == expected,
        observed,
    )


# ============================================================
# LOAD FROZEN CORPUS
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("CORPUS AND ASSIGNMENT AUDIT")
print("=" * 80)

pos = pd.read_csv(
    POS_PATH,
    sep="\t",
)

neg = pd.read_csv(
    NEG_PATH,
    sep="\t",
)

assignment = pd.read_csv(
    ASSIGNMENT_PATH,
    sep="\t",
)

groups = pd.read_csv(
    GROUPS_PATH,
    sep="\t",
)


for df in [
    pos,
    neg,
    assignment,
]:
    df[
        "protein_id"
    ] = df[
        "protein_id"
    ].astype(str)

pos[
    "label"
] = pos[
    "label"
].astype(int)

neg[
    "label"
] = neg[
    "label"
].astype(int)

assignment[
    "label"
] = assignment[
    "label"
].astype(int)

assignment[
    "length"
] = assignment[
    "length"
].astype(int)


check(
    "Positive source count",
    len(pos)
    == EXPECTED_POS,
    str(len(pos)),
)

check(
    "Negative source count",
    len(neg)
    == EXPECTED_NEG,
    str(len(neg)),
)

check(
    "Positive IDs unique",
    pos[
        "protein_id"
    ].is_unique,
)

check(
    "Negative IDs unique",
    neg[
        "protein_id"
    ].is_unique,
)

check(
    "No positive/negative ID overlap",
    set(
        pos[
            "protein_id"
        ]
    ).isdisjoint(
        set(
            neg[
                "protein_id"
            ]
        )
    ),
)

check(
    "Assignment row count",
    len(assignment)
    == EXPECTED_TOTAL,
    str(
        len(assignment)
    ),
)

check(
    "Assignment IDs unique",
    assignment[
        "protein_id"
    ].is_unique,
)


source_basic = pd.concat(
    [
        pos[
            [
                "protein_id",
                "sequence",
                "length",
                "label",
            ]
        ],
        neg[
            [
                "protein_id",
                "sequence",
                "length",
                "label",
            ]
        ],
    ],
    ignore_index=True,
)

source_basic[
    "length"
] = source_basic[
    "length"
].astype(int)

source_basic[
    "label"
] = source_basic[
    "label"
].astype(int)

check(
    "Frozen corpus total",
    len(source_basic)
    == EXPECTED_TOTAL,
)

check(
    "Assignment covers frozen corpus exactly",
    set(
        assignment[
            "protein_id"
        ]
    )
    == set(
        source_basic[
            "protein_id"
        ]
    ),
)


assignment_source = assignment.merge(
    source_basic,
    on="protein_id",
    how="left",
    validate="one_to_one",
    suffixes=(
        "_assignment",
        "_source",
    ),
)

check(
    "Assignment labels match frozen corpus",
    (
        assignment_source[
            "label_assignment"
        ]
        == assignment_source[
            "label_source"
        ]
    ).all(),
)

check(
    "Assignment lengths match frozen corpus",
    (
        assignment_source[
            "length_assignment"
        ]
        == assignment_source[
            "length_source"
        ]
    ).all(),
)


# ============================================================
# STRICT HOMOLOGY COMPONENT EVIDENCE
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("STRICT COMPONENT AUDIT")
print("=" * 80)

with tarfile.open(
    EVIDENCE_PATH,
    mode="r:gz",
) as tf:

    strict_members = [
        m
        for m in tf.getmembers()
        if (
            m.isfile()
            and m.name.endswith(
                "strict_homology_components.tsv"
            )
        )
    ]

    check(
        "Exactly one strict-component evidence table",
        len(strict_members)
        == 1,
        str(
            len(strict_members)
        ),
    )

    strict = pd.read_csv(
        tf.extractfile(
            strict_members[0]
        ),
        sep="\t",
    )


strict[
    "protein_id"
] = strict[
    "protein_id"
].astype(str)

strict[
    "label"
] = strict[
    "label"
].astype(int)

strict[
    "length"
] = strict[
    "length"
].astype(int)

strict[
    "homology_component"
] = strict[
    "homology_component"
].astype(str)


check(
    "Strict evidence row count",
    len(strict)
    == EXPECTED_TOTAL,
)

check(
    "Strict evidence IDs unique",
    strict[
        "protein_id"
    ].is_unique,
)

check(
    "Strict evidence covers corpus exactly",
    set(
        strict[
            "protein_id"
        ]
    )
    == set(
        source_basic[
            "protein_id"
        ]
    ),
)


assignment_strict = assignment.merge(
    strict[
        [
            "protein_id",
            "label",
            "length",
            "old_cdhit_cluster",
            "homology_component",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
    suffixes=(
        "_assignment",
        "_strict",
    ),
)

check(
    "Final strict components match archived evidence",
    (
        assignment_strict[
            "homology_component_assignment"
        ].astype(str)
        == assignment_strict[
            "homology_component_strict"
        ].astype(str)
    ).all(),
)

check(
    "Final old CD-HIT clusters match archived evidence",
    (
        assignment_strict[
            "old_cdhit_cluster_assignment"
        ].astype(str)
        == assignment_strict[
            "old_cdhit_cluster_strict"
        ].astype(str)
    ).all(),
)


component_stats = (
    strict.groupby(
        "homology_component",
        sort=True,
    )
    .agg(
        proteins=(
            "protein_id",
            "size",
        ),
        positives=(
            "label",
            "sum",
        ),
    )
    .reset_index()
)

component_stats[
    "negatives"
] = (
    component_stats[
        "proteins"
    ]
    - component_stats[
        "positives"
    ]
)

component_stats[
    "kind"
] = [
    component_kind(
        int(p),
        int(n),
    )
    for p, n
    in zip(
        component_stats[
            "positives"
        ],
        component_stats[
            "negatives"
        ],
    )
]


check(
    "Strict component total",
    len(component_stats)
    == EXPECTED_COMPONENTS,
    str(
        len(component_stats)
    ),
)

kind_counts = Counter(
    component_stats[
        "kind"
    ]
)

check(
    "Positive-only component count",
    kind_counts[
        "positive_only"
    ]
    == EXPECTED_POS_ONLY_COMPONENTS,
)

check(
    "Negative-only component count",
    kind_counts[
        "negative_only"
    ]
    == EXPECTED_NEG_ONLY_COMPONENTS,
)

check(
    "Mixed component count",
    kind_counts[
        "mixed"
    ]
    == EXPECTED_MIXED_COMPONENTS,
)


component_size = dict(
    zip(
        component_stats[
            "homology_component"
        ],
        component_stats[
            "proteins"
        ].astype(int),
    )
)

component_kind_map = dict(
    zip(
        component_stats[
            "homology_component"
        ],
        component_stats[
            "kind"
        ],
    )
)

component_to_proteins = {
    comp:
        list(
            g[
                "protein_id"
            ]
        )
    for comp, g
    in strict.groupby(
        "homology_component",
        sort=False,
    )
}


component_partitions = (
    assignment.groupby(
        "homology_component"
    )[
        "partition"
    ]
    .nunique()
)

check(
    "Zero strict-component partition overlap",
    (
        component_partitions
        == 1
    ).all(),
)


component_partition = (
    assignment.groupby(
        "homology_component"
    )[
        "partition"
    ]
    .first()
    .to_dict()
)

mixed_components = set(
    component_stats.loc[
        component_stats[
            "kind"
        ]
        == "mixed",
        "homology_component",
    ]
)

check(
    "All mixed components are train-only",
    all(
        component_partition[
            c
        ]
        == "train"
        for c
        in mixed_components
    ),
)


# ============================================================
# SPLIT-FILE CONTENT AUDIT
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("SPLIT FILE AUDIT")
print("=" * 80)

split_paths = {
    "train":
        TRAIN_PATH,

    "validation":
        VALIDATION_PATH,

    "test":
        TEST_PATH,

    "ood":
        OOD_PATH,
}

loaded_splits = {}

for partition, path in split_paths.items():

    df = pd.read_csv(
        path,
        sep="\t",
    )

    loaded_splits[
        partition
    ] = df

    required_cols = {
        "protein_id",
        "sequence",
        "length",
        "label",
    }

    check(
        f"{partition}: required columns",
        required_cols.issubset(
            set(
                df.columns
            )
        ),
    )

    check(
        f"{partition}: protein IDs unique",
        df[
            "protein_id"
        ].is_unique,
    )

    expected_ids = set(
        assignment.loc[
            assignment[
                "partition"
            ]
            == partition,
            "protein_id",
        ]
    )

    observed_ids = set(
        df[
            "protein_id"
        ].astype(str)
    )

    check(
        f"{partition}: IDs match assignment",
        observed_ids
        == expected_ids,
    )

    verified = df[
        [
            "protein_id",
            "sequence",
            "length",
            "label",
        ]
    ].merge(
        source_basic,
        on="protein_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "_split",
            "_source",
        ),
    )

    check(
        f"{partition}: sequences match frozen source",
        (
            verified[
                "sequence_split"
            ].astype(str)
            == verified[
                "sequence_source"
            ].astype(str)
        ).all(),
    )

    check(
        f"{partition}: labels match frozen source",
        (
            verified[
                "label_split"
            ].astype(int)
            == verified[
                "label_source"
            ].astype(int)
        ).all(),
    )

    check(
        f"{partition}: lengths match frozen source",
        (
            verified[
                "length_split"
            ].astype(int)
            == verified[
                "length_source"
            ].astype(int)
        ).all(),
    )


partition_counts = (
    assignment.groupby(
        [
            "partition",
            "label",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

expected_partition_counts = {
    "train":
        {
            0: 6076,
            1: 6076,
        },

    "validation":
        {
            0: 750,
            1: 750,
        },

    "test":
        {
            0: 750,
            1: 750,
        },

    "ood":
        {
            0: 321,
            1: 321,
        },
}

for partition, counts in expected_partition_counts.items():
    for label, expected in counts.items():

        observed = int(
            partition_counts.loc[
                partition,
                label,
            ]
        )

        check(
            (
                f"{partition} label={label} "
                f"count"
            ),
            observed
            == expected,
            str(
                observed
            ),
        )


# ============================================================
# AMRFINDER HIERARCHY RECONSTRUCTION
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("AMRFINDER BIOLOGICAL RECONSTRUCTION")
print("=" * 80)

catalog = pd.read_csv(
    AMRF_CATALOG_PATH,
    sep="\t",
    dtype=str,
)

hierarchy = pd.read_csv(
    AMRF_HIERARCHY_PATH,
    sep="\t",
    dtype=str,
)


node_set = {
    clean(x)
    for x
    in hierarchy[
        "node_id"
    ]
}

node_set.discard(
    None
)

parent_map = {}

for row in hierarchy.itertuples(
    index=False
):
    node = clean(
        row.node_id
    )

    parent = clean(
        row.parent_node_id
    )

    if node is not None:
        parent_map[
            node
        ] = parent


STRUCTURAL_ROOT_NODES = {
    node
    for node
    in node_set
    if (
        parent_map.get(
            node
        )
        is None
        or parent_map.get(
            node
        )
        not in node_set
        or parent_map.get(
            node
        )
        == node
    )
}

# Frozen biological-design rule:
#
# "AMR" is a broad ontology umbrella and is not considered
# a meaningful biological coverage anchor. Treat it as a
# non-informative root together with the structural hierarchy
# root(s).
ROOT_NODES = (
    set(
        STRUCTURAL_ROOT_NODES
    )
    | {
        "AMR",
    }
)

assert STRUCTURAL_ROOT_NODES == {
    "ALL",
}, (
    "Unexpected AMRFinder hierarchy structural roots: "
    f"{sorted(STRUCTURAL_ROOT_NODES)}"
)

assert ROOT_NODES == {
    "ALL",
    "AMR",
}


check(
    "Hierarchy/non-informative roots",
    len(ROOT_NODES)
    > 0,
    ", ".join(
        sorted(
            ROOT_NODES
        )
    ),
)


def ancestor_chain(node):
    node = clean(
        node
    )

    if node is None:
        return []

    chain = []
    seen = set()

    current = node

    while (
        current is not None
        and current not in seen
    ):
        chain.append(
            current
        )

        seen.add(
            current
        )

        parent = parent_map.get(
            current
        )

        if (
            parent is None
            or parent == current
        ):
            break

        current = parent

    return chain


depth_cache = {}


def node_depth(node):
    if node not in depth_cache:
        depth_cache[
            node
        ] = len(
            ancestor_chain(
                node
            )
        )

    return depth_cache[
        node
    ]


def lca_for_nodes(nodes):
    nodes = [
        n
        for n
        in nodes
        if clean(
            n
        )
        is not None
    ]

    if len(nodes) == 0:
        return None

    ancestor_sets = [
        set(
            ancestor_chain(
                n
            )
        )
        for n
        in nodes
    ]

    common = set.intersection(
        *ancestor_sets
    )

    if not common:
        return None

    return max(
        common,
        key=lambda n: (
            node_depth(
                n
            ),
            str(
                n
            ),
        ),
    )


amrf_acc_nodes = defaultdict(
    set
)

amrf_acc_classes = defaultdict(
    set
)

amrf_acc_subclasses = defaultdict(
    set
)

for _, row in catalog.iterrows():

    node = clean(
        row.get(
            "hierarchy_node"
        )
    )

    cls = clean(
        row.get(
            "class"
        )
    )

    subclass = clean(
        row.get(
            "subclass"
        )
    )

    # Match the builder's catalog eligibility: a record without a
    # hierarchy_node does NOT enter its accession lookup, including
    # the class/subclass categories attached to that record.
    if node is None:
        continue

    for column in [
        "refseq_protein_accession",
        "genbank_protein_accession",
    ]:

        accession = clean(
            row.get(
                column
            )
        )

        if accession is None:
            continue

        accession = norm_acc(
            accession
        )

        if accession is None:
            continue

        amrf_acc_nodes[
            accession
        ].add(
            node
        )

        if cls is not None:
            amrf_acc_classes[
                accession
            ].add(
                cls
            )

        if subclass is not None:
            amrf_acc_subclasses[
                accession
            ].add(
                subclass
            )


protein_amrf_nodes = {}
protein_amrf_classes = {}
protein_amrf_subclasses = {}

for row in pos.itertuples(
    index=False
):

    accessions = split_accessions(
        row.source_accessions
    )

    nodes = set()
    classes = set()
    subclasses = set()

    for acc in accessions:

        acc = norm_acc(
            acc
        )

        if acc is None:
            continue

        nodes |= amrf_acc_nodes.get(
            acc,
            set(),
        )

        classes |= amrf_acc_classes.get(
            acc,
            set(),
        )

        subclasses |= amrf_acc_subclasses.get(
            acc,
            set(),
        )

    protein_amrf_nodes[
        row.protein_id
    ] = nodes

    protein_amrf_classes[
        row.protein_id
    ] = classes

    protein_amrf_subclasses[
        row.protein_id
    ] = subclasses


amrf_mapped_positive_count = sum(
    bool(
        protein_amrf_nodes.get(
            pid,
            set(),
        )
    )
    for pid
    in pos[
        "protein_id"
    ]
)

check(
    "AMRF-mapped positive proteins",
    amrf_mapped_positive_count
    == EXPECTED_AMRF_MAPPED_POSITIVES,
    str(
        amrf_mapped_positive_count
    ),
)


pure_positive_components = set(
    component_stats.loc[
        component_stats[
            "kind"
        ]
        == "positive_only",
        "homology_component",
    ]
)


component_amrf_nodes = {}
component_amrf_mapped_count = {}
component_lca = {}

for comp in pure_positive_components:

    proteins = component_to_proteins[
        comp
    ]

    nodes = set()

    mapped_n = 0

    for pid in proteins:

        pnodes = protein_amrf_nodes.get(
            pid,
            set(),
        )

        if pnodes:
            mapped_n += 1
            nodes |= pnodes

    component_amrf_nodes[
        comp
    ] = nodes

    component_amrf_mapped_count[
        comp
    ] = mapped_n

    component_lca[
        comp
    ] = lca_for_nodes(
        sorted(
            nodes
        )
    )


amrf_node_to_components = defaultdict(
    set
)

for comp in pure_positive_components:

    all_ancestors = set()

    for node in component_amrf_nodes[
        comp
    ]:
        all_ancestors |= set(
            ancestor_chain(
                node
            )
        )

    for anc in all_ancestors:
        amrf_node_to_components[
            anc
        ].add(
            comp
        )


node_component_count = {
    node:
        len(
            comps
        )
    for node, comps
    in amrf_node_to_components.items()
}


def coverage_anchor(lca):
    if lca is None:
        return None

    for node in ancestor_chain(
        lca
    ):
        if node in ROOT_NODES:
            continue

        if (
            node_component_count.get(
                node,
                0,
            )
            >= 3
        ):
            return node

    return None


amrf_anchor = {
    comp:
        coverage_anchor(
            component_lca[
                comp
            ]
        )
    for comp
    in pure_positive_components
}

amrf_anchor_count = sum(
    value is not None
    for value
    in amrf_anchor.values()
)

check(
    "AMRF primary-anchor component count",
    amrf_anchor_count
    == EXPECTED_AMRF_ANCHORED_COMPONENTS,
    str(
        amrf_anchor_count
    ),
)


mapping_fraction = {
    comp:
        (
            component_amrf_mapped_count[
                comp
            ]
            / component_size[
                comp
            ]
        )
    for comp
    in pure_positive_components
}


zero_amrf_components = {
    comp
    for comp
    in pure_positive_components
    if component_amrf_mapped_count[
        comp
    ]
    == 0
}

check(
    "Zero-AMRF component count",
    len(
        zero_amrf_components
    )
    == EXPECTED_ZERO_AMRF_COMPONENTS,
    str(
        len(
            zero_amrf_components
        )
    ),
)


# ============================================================
# CARD RECONSTRUCTION
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("CARD BIOLOGICAL RECONSTRUCTION")
print("=" * 80)

card_acc_to_aro = {}

with CARD_FASTA_PATH.open(
    "r",
    encoding="utf-8",
    errors="replace",
) as f:

    for line in f:

        if not line.startswith(
            ">"
        ):
            continue

        m = re.match(
            r"^>[^|]+\|([^|]+)\|ARO:(\d+)\|",
            line,
        )

        if m is None:
            continue

        accession = m.group(
            1
        ).strip()

        aro = m.group(
            2
        ).strip()

        accession = norm_acc(
            accession
        )

        if accession is None:
            continue

        existing = card_acc_to_aro.get(
            accession
        )

        if (
            existing is not None
            and existing != aro
        ):
            raise AssertionError(
                "CARD FASTA accession mapped "
                "to multiple ARO IDs: "
                f"{accession}"
            )

        card_acc_to_aro[
            accession
        ] = aro


card_json = json.loads(
    CARD_JSON_PATH.read_text(
        encoding="utf-8"
    )
)

aro_families = defaultdict(
    set
)

aro_mechanisms = defaultdict(
    set
)

aro_drugs = defaultdict(
    set
)


def category_records(value):
    if isinstance(
        value,
        dict,
    ):
        return [
            v
            for v
            in value.values()
            if isinstance(
                v,
                dict,
            )
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            v
            for v
            in value
            if isinstance(
                v,
                dict,
            )
        ]

    return []


# The builder enumerates numeric CARD model keys, not metadata entries.
for model_key, entry in card_json.items():

    if not str(model_key).isdigit():
        continue

    if not isinstance(
        entry,
        dict,
    ):
        continue

    aro = normalize_aro(
        entry.get(
            "ARO_accession"
        )
    )

    if aro is None:
        continue

    for category in category_records(
        entry.get(
            "ARO_category",
            {},
        )
    ):

        class_name = clean(
            category.get(
                "category_aro_class_name"
            )
        )

        category_name = clean(
            category.get(
                "category_aro_name"
            )
        )

        if (
            class_name is None
            or category_name is None
        ):
            continue

        # The builder matches the explicit CARD annotation categories.
        class_key = class_name

        if class_key == "AMR Gene Family":
            aro_families[
                aro
            ].add(
                category_name
            )

        elif class_key == "Resistance Mechanism":
            aro_mechanisms[
                aro
            ].add(
                category_name
            )

        elif class_key == "Drug Class":
            aro_drugs[
                aro
            ].add(
                category_name
            )


protein_card_aros = {}
protein_card_families = {}
protein_card_mechanisms = {}
protein_card_drugs = {}

for row in pos.itertuples(
    index=False
):

    aros = set()

    # Builder: only proteins whose provenance contains CARD enter
    # all_card_protein / its category aggregation.
    if "CARD" in str(row.sources):
        raw_accessions = row.source_accessions
        if pd.notna(raw_accessions):
            tokens = {
                token.strip()
                for token in str(raw_accessions).split(";")
                if token.strip()
            }
        else:
            tokens = set()
    else:
        tokens = set()

    for token in tokens:

        accession_norm = norm_acc(
            token
        )

        if (
            accession_norm is not None
            and accession_norm
            in card_acc_to_aro
        ):
            aros.add(
                card_acc_to_aro[
                    accession_norm
                ]
            )

    # No direct ARO-token fallback: the frozen builder requires a
    # match through the CARD protein-homolog FASTA accession index.

    families = set()
    mechanisms = set()
    drugs = set()

    for aro in aros:
        families |= aro_families.get(
            aro,
            set(),
        )

        mechanisms |= aro_mechanisms.get(
            aro,
            set(),
        )

        drugs |= aro_drugs.get(
            aro,
            set(),
        )

    protein_card_aros[
        row.protein_id
    ] = aros

    protein_card_families[
        row.protein_id
    ] = families

    protein_card_mechanisms[
        row.protein_id
    ] = mechanisms

    protein_card_drugs[
        row.protein_id
    ] = drugs


card_supported_ids = set(
    pos.loc[
        pos[
            "sources"
        ].astype(str)
        .str.contains(
            "CARD",
            regex=False,
        ),
        "protein_id",
    ]
)

check(
    "CARD-supported positive count",
    len(
        card_supported_ids
    )
    == 5466,
    str(
        len(
            card_supported_ids
        )
    ),
)

card_mapped_supported = {
    pid
    for pid
    in card_supported_ids
    if protein_card_aros.get(
        pid,
        set(),
    )
}

check(
    "All CARD-supported positives map to CARD FASTA ARO",
    len(
        card_mapped_supported
    )
    == len(
        card_supported_ids
    ),
    (
        f"{len(card_mapped_supported)}/"
        f"{len(card_supported_ids)}"
    ),
)


def common_across_component(
    comp,
    mapping,
):
    proteins = component_to_proteins[
        comp
    ]

    sets = [
        set(
            mapping.get(
                pid,
                set(),
            )
        )
        for pid
        in proteins
    ]

    if not sets:
        return set()

    common = set(
        sets[0]
    )

    for values in sets[
        1:
    ]:
        common &= values

    return common


common_card_families = {
    comp:
        common_across_component(
            comp,
            protein_card_families,
        )
    for comp
    in zero_amrf_components
}

common_card_mechanisms = {
    comp:
        common_across_component(
            comp,
            protein_card_mechanisms,
        )
    for comp
    in zero_amrf_components
}


family_component_count = Counter()

for comp, families in common_card_families.items():
    for family in families:
        family_component_count[
            family
        ] += 1


mechanism_component_count = Counter()

for comp, mechanisms in common_card_mechanisms.items():
    for mechanism in mechanisms:
        mechanism_component_count[
            mechanism
        ] += 1


card_valid_family_anchors = {}
card_valid_mechanism_anchors = {}
card_valid_anchor_values = {}
card_anchor_kind = {}

for comp in zero_amrf_components:

    valid_families = {
        family
        for family
        in common_card_families[
            comp
        ]
        if family_component_count[
            family
        ]
        >= 3
    }

    valid_mechanisms = {
        mechanism
        for mechanism
        in common_card_mechanisms[
            comp
        ]
        if mechanism_component_count[
            mechanism
        ]
        >= 3
    }

    card_valid_family_anchors[
        comp
    ] = valid_families

    card_valid_mechanism_anchors[
        comp
    ] = valid_mechanisms

    if valid_families:
        card_anchor_kind[
            comp
        ] = "family"

        card_valid_anchor_values[
            comp
        ] = valid_families

    elif valid_mechanisms:
        card_anchor_kind[
            comp
        ] = "mechanism"

        card_valid_anchor_values[
            comp
        ] = valid_mechanisms

    else:
        card_anchor_kind[
            comp
        ] = None

        card_valid_anchor_values[
            comp
        ] = set()


card_family_anchor_count = sum(
    value == "family"
    for value
    in card_anchor_kind.values()
)

card_mechanism_anchor_count = sum(
    value == "mechanism"
    for value
    in card_anchor_kind.values()
)

card_none_count = sum(
    value is None
    for value
    in card_anchor_kind.values()
)


check(
    "CARD family-anchor component count",
    card_family_anchor_count
    == EXPECTED_CARD_FAMILY_ANCHORED_COMPONENTS,
    str(
        card_family_anchor_count
    ),
)

check(
    "CARD mechanism-anchor component count",
    card_mechanism_anchor_count
    == EXPECTED_CARD_MECHANISM_ANCHORED_COMPONENTS,
    str(
        card_mechanism_anchor_count
    ),
)

check(
    "CARD no-anchor component count",
    card_none_count
    == EXPECTED_CARD_UNANCHORED_COMPONENTS,
    str(
        card_none_count
    ),
)


card_anchored_components = {
    comp
    for comp
    in zero_amrf_components
    if card_anchor_kind[
        comp
    ]
    is not None
}

amrf_anchored_components = {
    comp
    for comp, anchor
    in amrf_anchor.items()
    if anchor is not None
}

primary_anchored_components = (
    amrf_anchored_components
    | card_anchored_components
)

primary_anchored_proteins = sum(
    component_size[
        comp
    ]
    for comp
    in primary_anchored_components
)


check(
    "Unified primary-anchor component count",
    len(
        primary_anchored_components
    )
    == EXPECTED_PRIMARY_ANCHORED_COMPONENTS,
    str(
        len(
            primary_anchored_components
        )
    ),
)

check(
    "Unified primary-anchor protein count",
    primary_anchored_proteins
    == EXPECTED_PRIMARY_ANCHORED_PROTEINS,
    str(
        primary_anchored_proteins
    ),
)


# ============================================================
# GENERATED BIOLOGICAL-GROUP METADATA VS INDEPENDENT REBUILD
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("BIOLOGICAL GROUP METADATA AUDIT")
print("=" * 80)

groups[
    "homology_component"
] = groups[
    "homology_component"
].astype(str)

check(
    "Biological-group table component coverage",
    set(
        groups[
            "homology_component"
        ]
    )
    == set(
        component_stats[
            "homology_component"
        ]
    ),
)


groups_lookup = groups.set_index(
    "homology_component"
)

for comp in pure_positive_components:

    generated_lca = clean(
        groups_lookup.loc[
            comp,
            "component_lca",
        ]
    )

    reconstructed_lca = clean(
        component_lca[
            comp
        ]
    )

    if generated_lca != reconstructed_lca:
        raise AssertionError(
            "Component LCA mismatch for "
            f"{comp}: generated={generated_lca!r}, "
            f"reconstructed={reconstructed_lca!r}"
        )

check(
    "Generated component LCAs match independent reconstruction",
    True,
)


for comp in pure_positive_components:

    generated_source = clean(
        groups_lookup.loc[
            comp,
            "primary_anchor_source",
        ]
    )

    generated_anchor = clean(
        groups_lookup.loc[
            comp,
            "primary_anchor",
        ]
    )

    if comp in amrf_anchored_components:

        if (
            generated_source
            != "AMRFinderPlus"
        ):
            raise AssertionError(
                f"{comp}: expected AMRFinderPlus anchor source."
            )

        if (
            generated_anchor
            != amrf_anchor[
                comp
            ]
        ):
            raise AssertionError(
                f"{comp}: AMRF anchor mismatch. "
                f"generated={generated_anchor!r}, "
                f"reconstructed={amrf_anchor[comp]!r}"
            )

    elif comp in card_anchored_components:

        if generated_source != "CARD":
            raise AssertionError(
                f"{comp}: expected CARD anchor source."
            )

        if (
            generated_anchor
            not in card_valid_anchor_values[
                comp
            ]
        ):
            raise AssertionError(
                f"{comp}: generated CARD anchor "
                f"{generated_anchor!r} is not independently valid."
            )

    else:

        if generated_anchor is not None:
            raise AssertionError(
                f"{comp}: expected no primary anchor, "
                f"found {generated_anchor!r}."
            )


check(
    "Generated primary anchors pass independent reconstruction",
    True,
)


# ============================================================
# TRANSITIVE OOD RECONSTRUCTION
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("TRANSITIVE OOD RECONSTRUCTION")
print("=" * 80)

adj = {
    comp:
        set()
    for comp
    in pure_positive_components
}

amrf_edges = set()
card_edges = set()


def connect(
    a,
    b,
    edge_store=None,
):
    if a == b:
        return

    adj[
        a
    ].add(
        b
    )

    adj[
        b
    ].add(
        a
    )

    if edge_store is not None:
        edge_store.add(
            tuple(
                sorted(
                    (
                        a,
                        b,
                    )
                )
            )
        )


for comp in pure_positive_components:

    lca = component_lca[
        comp
    ]

    if lca is None:
        continue

    if lca in ROOT_NODES:
        continue

    if (
        mapping_fraction[
            comp
        ]
        < 0.80
    ):
        continue

    subtree = (
        set(
            amrf_node_to_components.get(
                lca,
                set(),
            )
        )
        & pure_positive_components
    )

    for other in subtree:
        connect(
            comp,
            other,
            amrf_edges,
        )


component_card_families = {}

for comp in pure_positive_components:

    families = set()

    for pid in component_to_proteins[
        comp
    ]:
        families |= protein_card_families.get(
            pid,
            set(),
        )

    component_card_families[
        comp
    ] = families


family_to_components = defaultdict(
    set
)

for comp, families in component_card_families.items():
    for family in families:
        family_to_components[
            family
        ].add(
            comp
        )


for family, comps in family_to_components.items():

    for a, b in itertools.combinations(
        sorted(
            comps
        ),
        2,
    ):
        connect(
            a,
            b,
            card_edges,
        )


amrf_seed_components = {
    comp
    for comp
    in pure_positive_components
    if (
        comp
        not in primary_anchored_components
        and component_lca[
            comp
        ]
        is not None
        and component_lca[
            comp
        ]
        not in ROOT_NODES
        and mapping_fraction[
            comp
        ]
        >= 0.80
    )
}


card_seed_components = {
    comp
    for comp
    in zero_amrf_components
    if (
        comp
        not in primary_anchored_components
        and len(
            common_card_families[
                comp
            ]
        )
        > 0
    )
}


check(
    "AMRF OOD seed count",
    len(
        amrf_seed_components
    )
    == EXPECTED_AMRF_SEEDS,
    str(
        len(
            amrf_seed_components
        )
    ),
)

check(
    "CARD OOD seed count",
    len(
        card_seed_components
    )
    == EXPECTED_CARD_SEEDS,
    str(
        len(
            card_seed_components
        )
    ),
)


all_seeds = (
    amrf_seed_components
    | card_seed_components
)


def graph_component(start):
    visited = {
        start
    }

    queue = deque(
        [
            start
        ]
    )

    while queue:

        node = queue.popleft()

        for neighbor in adj[
            node
        ]:

            if neighbor in visited:
                continue

            visited.add(
                neighbor
            )

            queue.append(
                neighbor
            )

    return visited


candidate_blocks = []
covered_seed_components = set()

for seed in sorted(
    all_seeds
):

    if seed in covered_seed_components:
        continue

    block = graph_component(
        seed
    )

    candidate_blocks.append(
        block
    )

    covered_seed_components |= block


eligible_ood_blocks = [
    block
    for block
    in candidate_blocks
    if not (
        block
        & primary_anchored_components
    )
]


reconstructed_ood_components = set().union(
    *eligible_ood_blocks
) if eligible_ood_blocks else set()


reconstructed_ood_positive_proteins = sum(
    component_size[
        comp
    ]
    for comp
    in reconstructed_ood_components
)


check(
    "OOD biological block count",
    len(
        eligible_ood_blocks
    )
    == EXPECTED_OOD_BLOCKS,
    str(
        len(
            eligible_ood_blocks
        )
    ),
)

check(
    "OOD positive strict-component count",
    len(
        reconstructed_ood_components
    )
    == EXPECTED_OOD_POS_COMPONENTS,
    str(
        len(
            reconstructed_ood_components
        )
    ),
)

check(
    "OOD positive protein count",
    reconstructed_ood_positive_proteins
    == EXPECTED_OOD_POS_PROTEINS,
    str(
        reconstructed_ood_positive_proteins
    ),
)

check(
    "OOD has zero primary-anchor overlap",
    reconstructed_ood_components
    .isdisjoint(
        primary_anchored_components
    ),
)


assigned_ood_positive_components = set(
    assignment.loc[
        (
            assignment[
                "partition"
            ]
            == "ood"
        )
        & (
            assignment[
                "label"
            ]
            == 1
        ),
        "homology_component",
    ]
)

check(
    "Assigned OOD positive components equal independent closure",
    assigned_ood_positive_components
    == reconstructed_ood_components,
)


manifest_block_sets = {
    frozenset(
        block[
            "components"
        ]
    )
    for block
    in manifest[
        "ood_blocks"
    ]
}

reconstructed_block_sets = {
    frozenset(
        block
    )
    for block
    in eligible_ood_blocks
}

check(
    "Manifest OOD block memberships equal independent closure",
    manifest_block_sets
    == reconstructed_block_sets,
)


# ============================================================
# PRIMARY EVALUATION ELIGIBILITY / TRAIN ANCHOR RETENTION
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("PRIMARY POSITIVE ELIGIBILITY AUDIT")
print("=" * 80)

eligible_positive_components = {
    comp
    for comp
    in primary_anchored_components
    if (
        comp
        not in reconstructed_ood_components
        and component_size[
            comp
        ]
        <= PRIMARY_COMPONENT_CAP
    )
}

eligible_positive_proteins = sum(
    component_size[
        comp
    ]
    for comp
    in eligible_positive_components
)


check(
    "Eligible positive component count",
    len(
        eligible_positive_components
    )
    == EXPECTED_ELIGIBLE_POS_COMPONENTS,
    str(
        len(
            eligible_positive_components
        )
    ),
)

check(
    "Eligible positive protein count",
    eligible_positive_proteins
    == EXPECTED_ELIGIBLE_POS_PROTEINS,
    str(
        eligible_positive_proteins
    ),
)


primary_eval_positive_components = set(
    assignment.loc[
        (
            assignment[
                "label"
            ]
            == 1
        )
        & (
            assignment[
                "partition"
            ].isin(
                [
                    "validation",
                    "test",
                ]
            )
        ),
        "homology_component",
    ]
)


check(
    "Every primary evaluation positive component is eligible",
    primary_eval_positive_components
    .issubset(
        eligible_positive_components
    ),
)


unanchored_nonood_components = (
    pure_positive_components
    - primary_anchored_components
    - reconstructed_ood_components
)

check(
    "Unanchored non-OOD positives are train-only",
    all(
        component_partition[
            comp
        ]
        == "train"
        for comp
        in unanchored_nonood_components
    ),
)


for comp in sorted(
    primary_eval_positive_components
):

    generated_source = clean(
        groups_lookup.loc[
            comp,
            "primary_anchor_source",
        ]
    )

    generated_anchor = clean(
        groups_lookup.loc[
            comp,
            "primary_anchor",
        ]
    )

    if generated_source == "AMRFinderPlus":

        peer_components = {
            other
            for other, anchor
            in amrf_anchor.items()
            if (
                anchor
                == generated_anchor
                and other
                != comp
            )
        }

    elif generated_source == "CARD":

        peer_components = {
            other
            for other
            in card_anchored_components
            if (
                other != comp
                and generated_anchor
                in card_valid_anchor_values[
                    other
                ]
            )
        }

    else:
        raise AssertionError(
            f"{comp}: unsupported primary-anchor source "
            f"{generated_source!r}"
        )

    has_train_peer = any(
        component_partition.get(
            peer
        )
        == "train"
        for peer
        in peer_components
    )

    if not has_train_peer:
        raise AssertionError(
            f"{comp}: no separate training component "
            f"for anchor {generated_anchor!r}"
        )


check(
    "Every validation/test biological anchor retained in training",
    True,
)


# ============================================================
# COMPONENT CAPS
# ============================================================

for partition in [
    "validation",
    "test",
]:

    comps = set(
        assignment.loc[
            assignment[
                "partition"
            ]
            == partition,
            "homology_component",
        ]
    )

    check(
        f"{partition}: all component sizes <=112",
        all(
            component_size[
                comp
            ]
            <= PRIMARY_COMPONENT_CAP
            for comp
            in comps
        ),
    )


ood_negative_components = set(
    assignment.loc[
        (
            assignment[
                "partition"
            ]
            == "ood"
        )
        & (
            assignment[
                "label"
            ]
            == 0
        ),
        "homology_component",
    ]
)

check(
    "OOD negative component cap <=48",
    all(
        component_size[
            comp
        ]
        <= OOD_NEGATIVE_COMPONENT_CAP
        for comp
        in ood_negative_components
    ),
)


# ============================================================
# POSITIVE SOURCE BALANCE
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("POSITIVE SOURCE BALANCE AUDIT")
print("=" * 80)

positive_with_component = pos.merge(
    strict[
        [
            "protein_id",
            "homology_component",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
)

eligible_positive_df = (
    positive_with_component[
        positive_with_component[
            "homology_component"
        ].isin(
            eligible_positive_components
        )
    ]
    .copy()
)

check(
    "Eligible positive dataframe size",
    len(
        eligible_positive_df
    )
    == EXPECTED_ELIGIBLE_POS_PROTEINS,
)


source_balance_report = {}

pool_source_fraction = (
    eligible_positive_df[
        "sources"
    ]
    .astype(str)
    .value_counts(
        normalize=True
    )
    .to_dict()
)

for partition in [
    "validation",
    "test",
]:

    partition_ids = set(
        assignment.loc[
            (
                assignment[
                    "partition"
                ]
                == partition
            )
            & (
                assignment[
                    "label"
                ]
                == 1
            ),
            "protein_id",
        ]
    )

    df = positive_with_component[
        positive_with_component[
            "protein_id"
        ].isin(
            partition_ids
        )
    ]

    observed_fraction = (
        df[
            "sources"
        ]
        .astype(str)
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    source_balance_report[
        partition
    ] = {}

    for source_name, pool_fraction in sorted(
        pool_source_fraction.items()
    ):

        observed = float(
            observed_fraction.get(
                source_name,
                0.0,
            )
        )

        delta = (
            observed
            - float(
                pool_fraction
            )
        )

        source_balance_report[
            partition
        ][
            source_name
        ] = {
            "eligible_pool_fraction":
                float(
                    pool_fraction
                ),

            "observed_fraction":
                observed,

            "delta":
                float(
                    delta
                ),
        }

        check(
            (
                f"{partition} source balance: "
                f"{source_name}"
            ),
            abs(
                delta
            )
            <= 0.05
            + 1e-12,
            f"delta={delta:+.8f}",
        )


# ============================================================
# RESISTANCE-CATEGORY AUDIT
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("RESISTANCE-CATEGORY AUDIT")
print("=" * 80)

protein_category_tokens = {}

for row in pos.itertuples(
    index=False
):

    pid = row.protein_id

    tokens = set()

    for value in protein_amrf_classes.get(
        pid,
        set(),
    ):
        tokens.add(
            f"AMRF_CLASS::{value}"
        )

    for value in protein_amrf_subclasses.get(
        pid,
        set(),
    ):
        tokens.add(
            f"AMRF_SUBCLASS::{value}"
        )

    for value in protein_card_drugs.get(
        pid,
        set(),
    ):
        tokens.add(
            f"CARD_DRUG::{value}"
        )

    for value in protein_card_mechanisms.get(
        pid,
        set(),
    ):
        tokens.add(
            f"CARD_MECHANISM::{value}"
        )

    protein_category_tokens[
        pid
    ] = tokens


eligible_pool_ids = set(
    eligible_positive_df[
        "protein_id"
    ]
)

eligible_id_to_component = dict(
    zip(
        eligible_positive_df[
            "protein_id"
        ],
        eligible_positive_df[
            "homology_component"
        ],
    )
)


token_protein_counts = Counter()
token_components = defaultdict(
    set
)

for pid in eligible_pool_ids:

    for token in protein_category_tokens.get(
        pid,
        set(),
    ):

        token_protein_counts[
            token
        ] += 1

        token_components[
            token
        ].add(
            eligible_id_to_component[
                pid
            ]
        )


common_category_tokens = sorted(
    token
    for token, protein_count
    in token_protein_counts.items()
    if (
        protein_count * 100
        >= EXPECTED_ELIGIBLE_POS_PROTEINS
        and len(
            token_components[
                token
            ]
        )
        >= 3
    )
)


manifest_common_categories = sorted(
    manifest[
        "biological_grouping"
    ][
        "common_categories"
    ]
)

print(
    "Common resistance-category token count:",
    len(
        common_category_tokens
    ),
)

print(
    "Manifest common-category token count:",
    len(
        manifest_common_categories
    ),
)

# Category *balance* was audit-only in the builder; the saved
# category *inventory* is frozen provenance and MUST match.
# Run this before costly BLAST replay: fail closed, with precise diffs.
observed_category_set = set(common_category_tokens)
manifest_category_set = set(manifest_common_categories)

check(
    "Frozen resistance-category token inventory matches builder manifest",
    common_category_tokens == manifest_common_categories,
    (
        "auditor_only="
        + repr(sorted(observed_category_set - manifest_category_set))
        + "; manifest_only="
        + repr(sorted(manifest_category_set - observed_category_set))
    ),
)

# Category counts in validation/test remain descriptive, not
# additional optimization constraints.


category_audit = {}

for token in common_category_tokens:

    category_audit[
        token
    ] = {}

    for partition in [
        "validation",
        "test",
    ]:

        ids = set(
            assignment.loc[
                (
                    assignment[
                        "partition"
                    ]
                    == partition
                )
                & (
                    assignment[
                        "label"
                    ]
                    == 1
                ),
                "protein_id",
            ]
        )

        count = sum(
            token
            in protein_category_tokens.get(
                pid,
                set(),
            )
            for pid
            in ids
        )

        category_audit[
            token
        ][
            partition
        ] = int(
            count
        )


check(
    "Resistance-category audit completed",
    True,
)


# ============================================================
# EXACT LENGTH KS AUDIT
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("EXACT LENGTH KS AUDIT")
print("=" * 80)

eligible_positive_lengths = (
    eligible_positive_df[
        "length"
    ]
    .astype(int)
    .to_numpy()
)


def partition_lengths(
    partition,
    label,
):
    return (
        assignment.loc[
            (
                assignment[
                    "partition"
                ]
                == partition
            )
            & (
                assignment[
                    "label"
                ]
                == label
            ),
            "length",
        ]
        .astype(int)
        .to_numpy()
    )


positive_validation = partition_lengths(
    "validation",
    1,
)

positive_test = partition_lengths(
    "test",
    1,
)

positive_train = partition_lengths(
    "train",
    1,
)


ks_report = {
    "positive_validation_vs_eligible_pool":
        exact_ks(
            positive_validation,
            eligible_positive_lengths,
        ),

    "positive_test_vs_eligible_pool":
        exact_ks(
            positive_test,
            eligible_positive_lengths,
        ),

    "positive_validation_vs_test":
        exact_ks(
            positive_validation,
            positive_test,
        ),

    "positive_train_vs_validation_report_only":
        exact_ks(
            positive_train,
            positive_validation,
        ),

    "positive_train_vs_test_report_only":
        exact_ks(
            positive_train,
            positive_test,
        ),
}


for partition in [
    "train",
    "validation",
    "test",
    "ood",
]:

    ks_report[
        f"{partition}_positive_vs_negative"
    ] = exact_ks(
        partition_lengths(
            partition,
            1,
        ),
        partition_lengths(
            partition,
            0,
        ),
    )


hard_ks_keys = [
    "positive_validation_vs_eligible_pool",
    "positive_test_vs_eligible_pool",
    "positive_validation_vs_test",
    "train_positive_vs_negative",
    "validation_positive_vs_negative",
    "test_positive_vs_negative",
    "ood_positive_vs_negative",
]


for key in hard_ks_keys:

    result = ks_report[
        key
    ]

    check(
        f"Exact KS <=0.10: {key}",
        result[
            "leq_0_10_exact"
        ],
        (
            f"KS={result['ks']:.12f}; "
            f"dev={result['cross_multiplied_deviation']}/"
            f"{result['denominator']}"
        ),
    )


# ============================================================
# FULL BLAST REPLAY: ORIGINAL RULE + STRONGER SYMMETRIC SCREEN
# ============================================================

print("\n" + "=" * 80)
print("FULL ARCHIVED BLAST REPLAY")
print("=" * 80)

protein_to_component = dict(zip(
    strict["protein_id"], strict["homology_component"]
))
protein_to_partition = dict(zip(
    assignment["protein_id"], assignment["partition"]
))


def classify_blast_alignment(qid, sid, nident, qlen, slen,
                             qstart, qend, sstart, send):
    """Independent integer-exact tests for two distinct BLAST rules.

    Historical: reproduce the frozen preprocessing algorithm literally:
      * discard self-alignment
      * qlen must be <= slen (the shorter sequence is QUERY)
      * for equal lengths, qid must be lexically smaller than sid
      * (qend - qstart + 1) / qlen >= 0.80
      * nident / qlen >= 0.40

    Broad: also check every reverse-directed row as an additional
    leak-detection safeguard. This is deliberately NOT compared against
    the historical undirected/canonical alignment count.
    """
    shorter = min(qlen, slen)
    shorter_span = (
        abs(qend - qstart) + 1
        if qlen <= slen
        else abs(send - sstart) + 1
    )
    broad = (
        shorter_span * 5 >= 4 * shorter
        and nident * 5 >= 2 * shorter
    )
    canonical_direction = (
        qid != sid
        and qlen <= slen
        and (qlen != slen or qid < sid)
    )
    historical = (
        canonical_direction
        and (abs(qend - qstart) + 1) * 5 >= 4 * qlen
        and nident * 5 >= 2 * qlen
    )
    return historical, broad


blast_total_rows = 0
blast_historical_qualifying_rows = 0
blast_qualifying_rows = 0  # broader directed screen, includes self
blast_qualifying_self_rows = 0
blast_unknown_qualifying_rows = 0
blast_cross_component_rows = 0
blast_cross_partition_rows = 0
blast_historical_cross_component_rows = 0
blast_historical_cross_partition_rows = 0
first_cross_component = None
first_cross_partition = None
first_unknown = None

with tarfile.open(EVIDENCE_PATH, mode="r:gz") as tf:
    blast_members = [
        member for member in tf.getmembers()
        if member.isfile() and member.name.endswith("all_vs_all.tsv")
    ]
    check(
        "Exactly one archived all-vs-all BLAST table",
        len(blast_members) == 1,
        str(len(blast_members)),
    )
    with tf.extractfile(blast_members[0]) as fh:
        for raw_line in fh:
            blast_total_rows += 1
            fields = raw_line.decode("utf-8", errors="strict").rstrip("\r\n").split("\t")
            if len(fields) != 12:
                raise AssertionError(
                    f"Unexpected BLAST column count at row {blast_total_rows}: {len(fields)}"
                )
            (
                qseqid, sseqid, pident, nident_raw, qlen_raw, slen_raw,
                qstart_raw, qend_raw, sstart_raw, send_raw,
                evalue, bitscore,
            ) = fields
            nident = int(nident_raw)
            qlen = int(qlen_raw)
            slen = int(slen_raw)
            qstart = int(qstart_raw)
            qend = int(qend_raw)
            sstart = int(sstart_raw)
            send = int(send_raw)
            if qlen <= 0 or slen <= 0 or nident < 0:
                raise AssertionError(f"Invalid BLAST sequence length/identity at row {blast_total_rows}")

            historical, broad = classify_blast_alignment(
                qseqid, sseqid, nident, qlen, slen,
                qstart, qend, sstart, send,
            )
            if historical and not broad:
                raise AssertionError("Historical BLAST qualifier excluded by broad test")

            if historical:
                blast_historical_qualifying_rows += 1
            if broad:
                blast_qualifying_rows += 1
                if qseqid == sseqid:
                    blast_qualifying_self_rows += 1
                if (
                    qseqid not in protein_to_component
                    or sseqid not in protein_to_component
                    or qseqid not in protein_to_partition
                    or sseqid not in protein_to_partition
                ):
                    blast_unknown_qualifying_rows += 1
                    if first_unknown is None:
                        first_unknown = (qseqid, sseqid)
                else:
                    q_comp = protein_to_component[qseqid]
                    s_comp = protein_to_component[sseqid]
                    if q_comp != s_comp:
                        blast_cross_component_rows += 1
                        if historical:
                            blast_historical_cross_component_rows += 1
                        if first_cross_component is None:
                            first_cross_component = {
                                "qseqid": qseqid, "sseqid": sseqid,
                                "q_component": q_comp, "s_component": s_comp,
                                "qlen": qlen, "slen": slen, "nident": nident,
                            }
                    q_partition = protein_to_partition[qseqid]
                    s_partition = protein_to_partition[sseqid]
                    if q_partition != s_partition:
                        blast_cross_partition_rows += 1
                        if historical:
                            blast_historical_cross_partition_rows += 1
                        if first_cross_partition is None:
                            first_cross_partition = {
                                "qseqid": qseqid, "sseqid": sseqid,
                                "q_partition": q_partition,
                                "s_partition": s_partition,
                                "qlen": qlen, "slen": slen, "nident": nident,
                            }
            if blast_total_rows % 2_000_000 == 0:
                print(
                    f"  scanned {blast_total_rows:,} | "
                    f"historical={blast_historical_qualifying_rows:,} | "
                    f"broader_directed={blast_qualifying_rows:,}",
                    flush=True,
                )

print("BLAST total rows:", f"{blast_total_rows:,}")
print("Original canonical qualifying BLAST rows:", f"{blast_historical_qualifying_rows:,}")
print("Broader directed qualifying BLAST rows:", f"{blast_qualifying_rows:,}")
print("Broader qualifying self rows:", f"{blast_qualifying_self_rows:,}")
print("Broader qualifying cross-component rows:", f"{blast_cross_component_rows:,}")
print("Broader qualifying cross-partition rows:", f"{blast_cross_partition_rows:,}")

check(
    "Complete archived BLAST row count",
    blast_total_rows == EXPECTED_ALL_VS_ALL_BLAST_ROWS,
    f"observed={blast_total_rows:,}; expected={EXPECTED_ALL_VS_ALL_BLAST_ROWS:,}",
)
check(
    "Original frozen BLAST canonical qualifying-pair count reproduced",
    blast_historical_qualifying_rows == EXPECTED_QUALIFYING_BLAST_ROWS,
    f"observed={blast_historical_qualifying_rows:,}; expected={EXPECTED_QUALIFYING_BLAST_ROWS:,}",
)
check(
    "Additional bidirectional BLAST qualifying-row count reproduced",
    blast_qualifying_rows == EXPECTED_BIDIRECTIONAL_QUALIFYING_ROWS,
    f"observed={blast_qualifying_rows:,}; expected={EXPECTED_BIDIRECTIONAL_QUALIFYING_ROWS:,}",
)
check(
    "BLAST qualifying self-alignments counted",
    blast_qualifying_self_rows == EXPECTED_BIDIRECTIONAL_SELF_ROWS,
    f"observed={blast_qualifying_self_rows:,}; expected={EXPECTED_BIDIRECTIONAL_SELF_ROWS:,}",
)
check(
    "Zero qualifying BLAST rows reference unknown proteins",
    blast_unknown_qualifying_rows == 0,
    str(first_unknown),
)
check(
    "Historical BLAST qualifying pairs never cross strict components",
    blast_historical_cross_component_rows == 0,
)
check(
    "Historical BLAST qualifying pairs never cross final partitions",
    blast_historical_cross_partition_rows == 0,
)
check(
    "Additional bidirectional BLAST screen has zero cross-component rows",
    blast_cross_component_rows == 0,
    str(first_cross_component),
)
check(
    "Additional bidirectional BLAST screen has zero cross-partition rows",
    blast_cross_partition_rows == 0,
    str(first_cross_partition),
)


# ============================================================
# FINAL MANIFEST CONSISTENCY
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("FINAL MANIFEST CONSISTENCY")
print("=" * 80)

check(
    "Manifest OOD block count",
    manifest[
        "biological_grouping"
    ][
        "ood_blocks"
    ]
    == EXPECTED_OOD_BLOCKS,
)

check(
    "Manifest OOD positive-component count",
    manifest[
        "biological_grouping"
    ][
        "ood_positive_components"
    ]
    == EXPECTED_OOD_POS_COMPONENTS,
)

check(
    "Manifest OOD positive-protein count",
    manifest[
        "biological_grouping"
    ][
        "ood_positive_proteins"
    ]
    == EXPECTED_OOD_POS_PROTEINS,
)

check(
    "Manifest primary-anchored component count",
    manifest[
        "biological_grouping"
    ][
        "primary_anchored_components"
    ]
    == EXPECTED_PRIMARY_ANCHORED_COMPONENTS,
)

check(
    "Manifest primary-anchored protein count",
    manifest[
        "biological_grouping"
    ][
        "primary_anchored_proteins"
    ]
    == EXPECTED_PRIMARY_ANCHORED_PROTEINS,
)


# ============================================================
# WRITE PASS REPORT ONLY AFTER EVERY HARD CHECK PASSES
# ============================================================

print(
    "\n"
    + "=" * 80
)
print("WRITING INDEPENDENT AUDIT REPORT")
print("=" * 80)

report = {
    "schema":
        "amr_binary_study_v2_independent_audit_v1",

    "status":
        "PASS",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "manifest_sha256_before_audit":
        manifest_hash,

    "audit_script_sha256":
        audit_script_hash,

    "design_note_sha256":
        sha256_file(
            DESIGN_NOTE_PATH
        ),

    "checks":
        CHECKS,

    "partition_counts":
        {
            partition:
                {
                    str(label):
                        int(
                            partition_counts.loc[
                                partition,
                                label,
                            ]
                        )
                    for label
                    in [
                        0,
                        1,
                    ]
                }
            for partition
            in [
                "train",
                "validation",
                "test",
                "ood",
            ]
        },

    "component_inventory": {
        "total":
            int(
                len(
                    component_stats
                )
            ),

        "positive_only":
            int(
                kind_counts[
                    "positive_only"
                ]
            ),

        "negative_only":
            int(
                kind_counts[
                    "negative_only"
                ]
            ),

        "mixed":
            int(
                kind_counts[
                    "mixed"
                ]
            ),
    },

    "biological_reconstruction": {
        "amrf_mapped_positive_proteins":
            int(
                amrf_mapped_positive_count
            ),

        "amrf_anchored_components":
            int(
                amrf_anchor_count
            ),

        "card_family_anchored_components":
            int(
                card_family_anchor_count
            ),

        "card_mechanism_anchored_components":
            int(
                card_mechanism_anchor_count
            ),

        "card_unanchored_zero_amrf_components":
            int(
                card_none_count
            ),

        "primary_anchored_components":
            int(
                len(
                    primary_anchored_components
                )
            ),

        "primary_anchored_proteins":
            int(
                primary_anchored_proteins
            ),

        "eligible_positive_components":
            int(
                len(
                    eligible_positive_components
                )
            ),

        "eligible_positive_proteins":
            int(
                eligible_positive_proteins
            ),
    },

    "ood_reconstruction": {
        "amrf_seed_components":
            int(
                len(
                    amrf_seed_components
                )
            ),

        "card_seed_components":
            int(
                len(
                    card_seed_components
                )
            ),

        "amrf_graph_edges":
            int(
                len(
                    amrf_edges
                )
            ),

        "card_family_graph_edges":
            int(
                len(
                    card_edges
                )
            ),

        "eligible_blocks":
            int(
                len(
                    eligible_ood_blocks
                )
            ),

        "positive_components":
            int(
                len(
                    reconstructed_ood_components
                )
            ),

        "positive_proteins":
            int(
                reconstructed_ood_positive_proteins
            ),

        "blocks":
            [
                sorted(
                    block
                )
                for block
                in sorted(
                    eligible_ood_blocks,
                    key=lambda block: (
                        -sum(
                            component_size[
                                c
                            ]
                            for c
                            in block
                        ),
                        min(
                            block
                        ),
                    ),
                )
            ],
    },

    "length_ks_exact":
        ks_report,

    "positive_source_balance":
        source_balance_report,

    "resistance_category_audit": {
        "common_category_count":
            int(
                len(
                    common_category_tokens
                )
            ),

        "common_categories":
            common_category_tokens,

        "matches_frozen_manifest":
            common_category_tokens == manifest_common_categories,

        "validation_test_counts":
            category_audit,
    },

    "blast_replay": {
        "original_canonical_rule": {
            "direction": "exclude self; shorter query only; equal lengths require qseqid < sseqid",
            "coverage": "query coordinate span / qlen >= 0.80",
            "identity": "nident / qlen >= 0.40",
        },
        "broader_bidirectional_rule": {
            "coverage": "shorter-sequence coordinate span / shorter length >= 0.80",
            "identity": "nident / shorter length >= 0.40",
        },
        "total_rows": int(blast_total_rows),
        "qualifying_rows": int(blast_historical_qualifying_rows),
        "qualifying_self_rows": 0,
        "qualifying_cross_component_rows": int(blast_historical_cross_component_rows),
        "qualifying_cross_partition_rows": int(blast_historical_cross_partition_rows),
        "bidirectional_qualifying_rows": int(blast_qualifying_rows),
        "bidirectional_qualifying_self_rows": int(blast_qualifying_self_rows),
        "bidirectional_qualifying_cross_component_rows": int(blast_cross_component_rows),
        "bidirectional_qualifying_cross_partition_rows": int(blast_cross_partition_rows),
    },

    "verified_output_hashes":
        observed_output_hashes,

    "held_out_predictions_generated": {
        "test":
            False,

        "ood":
            False,
    },
}


AUDIT_REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

tmp_path = None

try:

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=".v2_split_audit.",
        suffix=".tmp",
        dir=AUDIT_REPORT_PATH.parent,
        delete=False,
    ) as tmp:

        tmp_path = Path(
            tmp.name
        )

        json.dump(
            report,
            tmp,
            indent=2,
            sort_keys=True,
        )

        tmp.write(
            "\n"
        )

        tmp.flush()
        os.fsync(
            tmp.fileno()
        )

    os.replace(
        tmp_path,
        AUDIT_REPORT_PATH,
    )

    tmp_path = None

finally:

    if (
        tmp_path is not None
        and tmp_path.exists()
    ):
        tmp_path.unlink()


audit_report_hash = sha256_file(
    AUDIT_REPORT_PATH
)

print(
    "Audit report:",
    AUDIT_REPORT_PATH,
)

print(
    "Audit report SHA256:",
    audit_report_hash,
)


print(
    "\n"
    + "=" * 80
)

print(
    "INDEPENDENT SPLIT AUDIT: PASS"
)

print(
    "=" * 80
)

print(
    "Manifest was NOT modified."
)

print(
    "No model predictions were generated."
)
