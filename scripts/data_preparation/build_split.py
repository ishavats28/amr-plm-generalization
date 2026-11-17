
from __future__ import annotations

from pathlib import Path
from collections import defaultdict, Counter, deque
from fractions import Fraction
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import shutil
import tarfile
import tempfile
import re

import numpy as np
import pandas as pd
import ortools
from ortools.sat.python import cp_model


# ============================================================
# FROZEN STUDY CONSTANTS
# ============================================================

ROOT = Path(
    "."
)

POS_PATH = ROOT / "data/interim/positive_model_range.tsv.gz"
NEG_PATH = ROOT / "data/interim/negative_model_range.tsv.gz"

EVIDENCE_PATH = (
    ROOT / "metadata/preprocessing_evidence.tar.gz"
)

AMRF_CATALOG_PATH = (
    ROOT / "data/raw/amrfinderplus/ReferenceGeneCatalog.txt"
)

AMRF_HIERARCHY_PATH = (
    ROOT / "data/raw/amrfinderplus/ReferenceGeneHierarchy.txt"
)

CARD_FASTA_PATH = (
    ROOT / "data/raw/card/protein_fasta_protein_homolog_model.fasta"
)

CARD_JSON_PATH = (
    ROOT / "data/raw/card/card.json"
)

DESIGN_NOTE = ROOT / "MANUSCRIPT_NOTES_v2.md"

OUT_PROCESSED = ROOT / "data/processed"
OUT_METADATA = ROOT / "metadata"

TARGET_FILES = [
    OUT_PROCESSED / "train.tsv.gz",
    OUT_PROCESSED / "validation.tsv.gz",
    OUT_PROCESSED / "test.tsv.gz",
    OUT_PROCESSED / "ood.tsv.gz",
    OUT_METADATA / "v2_split_assignment.tsv.gz",
    OUT_METADATA / "v2_biological_groups.tsv.gz",
    OUT_METADATA / "v2_split_manifest.json",
]

EXPECTED_DESIGN_NOTE_SHA256 = (
    "6abba0b9095ff3b5243ace7f65680395a4430127ae6e3042f27765fd4b0ea149"
)

EXPECTED_POS_SHA256 = (
    "80a95666b286e3912ab65c7500a2d5b84406de870d93de8c85367665fed70089"
)

EXPECTED_NEG_SHA256 = (
    "4e40bbd40c6f2a31359f1ef39445701f2e76adc05346880e937a01dcf906a0b3"
)

EXPECTED_EVIDENCE_SHA256 = (
    "d3c7bb505e192c200d81483beeb87e0916001323c4b3b84482137c75f6cec671"
)

EXPECTED_HIERARCHY_SHA256 = (
    "7331c0fdd8a9dcb1400a043ffef84912728624fca0c2d5e0747a605ac09c8882"
)

EXPECTED_CARD_JSON_SHA256 = (
    "d9a0576e6068762b224c9d3b0f6ec76083e3bfd7323e0f73d1584ef8e2ae7c0f"
)

EXPECTED_AMRF_VERSION = "2026-08-07.1"
EXPECTED_CARD_VERSION = "4.0.2"

SPLIT_SEED = 20261019

VAL_POS_TARGET = 750
TEST_POS_TARGET = 750
OOD_POS_EXPECTED = 321
TRAIN_POS_EXPECTED = 6076

VAL_NEG_TARGET = 750
TEST_NEG_TARGET = 750
OOD_NEG_TARGET = 321
TRAIN_NEG_EXPECTED = 6076

PRIMARY_COMPONENT_CAP = 112
OOD_NEG_COMPONENT_CAP = 48

KS_LIMIT = Fraction(1, 10)

LENGTH_BINS = [
    (40, 100),
    (101, 200),
    (201, 400),
    (401, 600),
    (601, 800),
    (801, 1022),
]

ROOT_NODES = {"AMR", "ALL"}

COMMON_CATEGORY_MIN_FRACTION = Fraction(1, 100)
COMMON_CATEGORY_MIN_COMPONENTS = 3


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def norm_acc(x):
    if pd.isna(x):
        return None
    return re.sub(
        r"\.\d+$",
        "",
        str(x).strip(),
    )


def deterministic_rank(items, namespace):
    keyed = []
    for item in items:
        token = (
            f"{SPLIT_SEED}|{namespace}|{item}"
        ).encode()
        digest = hashlib.sha256(token).hexdigest()
        keyed.append((digest, item))

    keyed.sort()

    return {
        item: rank
        for rank, (_, item)
        in enumerate(keyed, start=1)
    }


def write_tsv_gz_deterministic(
    df: pd.DataFrame,
    path: Path,
):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 9,
            "mtime": 0,
        },
    )


def fraction_bounds(
    numerator: int,
    denominator: int,
    target_n: int,
    tolerance: Fraction,
):
    p = Fraction(
        numerator,
        denominator,
    )

    low = max(
        Fraction(0),
        p - tolerance,
    )

    high = min(
        Fraction(1),
        p + tolerance,
    )

    low_n = math.ceil(
        low * target_n
    )

    high_n = math.floor(
        high * target_n
    )

    return low_n, high_n


def ks_statistic(a, b):
    a = np.sort(
        np.asarray(a, dtype=int)
    )
    b = np.sort(
        np.asarray(b, dtype=int)
    )

    values = np.unique(
        np.concatenate([a, b])
    )

    ia = np.searchsorted(
        a,
        values,
        side="right",
    )

    ib = np.searchsorted(
        b,
        values,
        side="right",
    )

    da = ia / len(a)
    db = ib / len(b)

    return float(
        np.max(
            np.abs(da - db)
        )
    )


def solve_optimal(
    model,
    objective,
    sense,
    label,
    timeout=180,
):
    if sense == "min":
        model.Minimize(objective)
    elif sense == "max":
        model.Maximize(objective)
    else:
        raise ValueError(sense)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = SPLIT_SEED
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    status_name = solver.StatusName(status)

    print(
        f"{label}: {status_name}"
    )

    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            f"{label} did not reach OPTIMAL status. "
            f"Status={status_name}"
        )

    optimum = int(
        round(
            solver.ObjectiveValue()
        )
    )

    print(
        f"{label} optimum: {optimum}"
    )

    model.Add(
        objective == optimum
    )

    return optimum


# ============================================================
# PRE-FLIGHT
# ============================================================

for p in [
    POS_PATH,
    NEG_PATH,
    EVIDENCE_PATH,
    AMRF_CATALOG_PATH,
    AMRF_HIERARCHY_PATH,
    CARD_FASTA_PATH,
    CARD_JSON_PATH,
    DESIGN_NOTE,
]:
    assert p.is_file(), p

for p in TARGET_FILES:
    assert not p.exists(), (
        f"Refusing to overwrite existing partition output: {p}"
    )

assert sha256_file(
    DESIGN_NOTE
) == EXPECTED_DESIGN_NOTE_SHA256

assert sha256_file(
    POS_PATH
) == EXPECTED_POS_SHA256

assert sha256_file(
    NEG_PATH
) == EXPECTED_NEG_SHA256

assert sha256_file(
    EVIDENCE_PATH
) == EXPECTED_EVIDENCE_SHA256

assert sha256_file(
    AMRF_HIERARCHY_PATH
) == EXPECTED_HIERARCHY_SHA256

assert sha256_file(
    CARD_JSON_PATH
) == EXPECTED_CARD_JSON_SHA256

print("Frozen input hashes: PASS")


# ============================================================
# LOAD CORPUS + STRICT COMPONENTS
# ============================================================

pos = pd.read_csv(
    POS_PATH,
    sep="\t",
)

neg = pd.read_csv(
    NEG_PATH,
    sep="\t",
)

assert len(pos) == 7897
assert len(neg) == 7897

with tarfile.open(
    EVIDENCE_PATH,
    "r:gz",
) as tar:
    f = tar.extractfile(
        "strict_homology_components.tsv"
    )
    assert f is not None

    components = pd.read_csv(
        f,
        sep="\t",
    )

assert len(components) == 15794
assert components["protein_id"].is_unique

corpus_basic = pd.concat(
    [
        pos[
            [
                "protein_id",
                "length",
                "label",
            ]
        ],
        neg[
            [
                "protein_id",
                "length",
                "label",
            ]
        ],
    ],
    ignore_index=True,
)

corpus_basic = corpus_basic.merge(
    components[
        [
            "protein_id",
            "old_cdhit_cluster",
            "homology_component",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
)

assert len(corpus_basic) == 15794
assert corpus_basic[
    "homology_component"
].notna().all()

component_stats = (
    corpus_basic.groupby(
        "homology_component"
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

component_stats["negatives"] = (
    component_stats["proteins"]
    - component_stats["positives"]
)

component_stats["kind"] = np.select(
    [
        component_stats["positives"].eq(
            component_stats["proteins"]
        ),
        component_stats["negatives"].eq(
            component_stats["proteins"]
        ),
    ],
    [
        "positive_only",
        "negative_only",
    ],
    default="mixed",
)

assert (
    component_stats["kind"]
    == "positive_only"
).sum() == 339

assert (
    component_stats["kind"]
    == "negative_only"
).sum() == 2339

assert (
    component_stats["kind"]
    == "mixed"
).sum() == 10

positive_only_components = set(
    component_stats.loc[
        component_stats["kind"]
        == "positive_only",
        "homology_component",
    ]
)

negative_only_components = set(
    component_stats.loc[
        component_stats["kind"]
        == "negative_only",
        "homology_component",
    ]
)

mixed_components = set(
    component_stats.loc[
        component_stats["kind"]
        == "mixed",
        "homology_component",
    ]
)

print("Strict component inventory: PASS")


# ============================================================
# AMRFINDERPLUS BIOLOGICAL STRUCTURE
# ============================================================

catalog = pd.read_csv(
    AMRF_CATALOG_PATH,
    sep="\t",
    dtype=str,
    low_memory=False,
)

hierarchy = pd.read_csv(
    AMRF_HIERARCHY_PATH,
    sep="\t",
    dtype=str,
    low_memory=False,
)

catalog_versions = sorted(
    set(
        catalog["db_version"]
        .dropna()
        .astype(str)
    )
)

hierarchy_versions = sorted(
    set(
        hierarchy["db_version"]
        .dropna()
        .astype(str)
    )
)

assert catalog_versions == [
    EXPECTED_AMRF_VERSION
]

assert hierarchy_versions == [
    EXPECTED_AMRF_VERSION
]

hier = hierarchy.drop_duplicates(
    subset=["node_id"]
).copy()

assert hier["node_id"].is_unique

parent = dict(
    zip(
        hier["node_id"],
        hier["parent_node_id"],
    )
)

hier_meta = hier.set_index(
    "node_id"
)


def ancestor_chain(node):
    chain = []
    seen = set()
    current = str(node)

    while (
        current
        and current not in seen
        and current in parent
    ):
        chain.append(current)
        seen.add(current)

        nxt = parent.get(current)

        if pd.isna(nxt):
            break

        nxt = str(nxt).strip()

        if not nxt:
            break

        current = nxt

    return chain


def node_depth(node):
    return len(
        ancestor_chain(node)
    )


amrf_lookup_rows = []

for _, row in catalog.iterrows():
    node = row.get(
        "hierarchy_node"
    )

    if pd.isna(node):
        continue

    for col in [
        "refseq_protein_accession",
        "genbank_protein_accession",
    ]:
        acc = row.get(col)

        if pd.isna(acc):
            continue

        amrf_lookup_rows.append({
            "accession_norm":
                norm_acc(acc),

            "hierarchy_node":
                str(node),

            "amrf_class":
                row.get("class"),

            "amrf_subclass":
                row.get("subclass"),
        })

amrf_lookup = (
    pd.DataFrame(
        amrf_lookup_rows
    )
    .drop_duplicates()
)

amrf_acc_rows = []

for row in pos.itertuples(
    index=False
):
    raw = row.source_accessions

    if pd.isna(raw):
        continue

    for token in str(raw).split(";"):
        token = token.strip()

        if not token:
            continue

        amrf_acc_rows.append({
            "protein_id":
                row.protein_id,

            "accession_norm":
                norm_acc(token),
        })

mapped_amrf = (
    pd.DataFrame(
        amrf_acc_rows
    )
    .merge(
        amrf_lookup,
        on="accession_norm",
        how="left",
    )
)

protein_node_counts = (
    mapped_amrf
    .dropna(
        subset=["hierarchy_node"]
    )
    .groupby("protein_id")[
        "hierarchy_node"
    ]
    .nunique()
)

assert (
    protein_node_counts <= 1
).all()

protein_nodes = (
    mapped_amrf
    .dropna(
        subset=["hierarchy_node"]
    )[
        [
            "protein_id",
            "hierarchy_node",
        ]
    ]
    .drop_duplicates("protein_id")
    .merge(
        components[
            [
                "protein_id",
                "homology_component",
            ]
        ],
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
)

assert (
    protein_nodes["protein_id"]
    .nunique()
    == 7568
)

component_node_lists = (
    protein_nodes[
        protein_nodes[
            "homology_component"
        ].isin(
            positive_only_components
        )
    ]
    .groupby(
        "homology_component"
    )["hierarchy_node"]
    .agg(
        lambda x:
            sorted(
                set(x)
            )
    )
)


def lowest_common_ancestor(nodes):
    nodes = list(
        dict.fromkeys(nodes)
    )

    if not nodes:
        return None

    chains = [
        ancestor_chain(node)
        for node in nodes
    ]

    common = set(
        chains[0]
    )

    for chain in chains[1:]:
        common &= set(chain)

    if not common:
        return None

    return max(
        common,
        key=node_depth,
    )


component_lca = {
    comp:
        lowest_common_ancestor(nodes)
    for comp, nodes
    in component_node_lists.items()
}

amrf_node_to_components = defaultdict(
    set
)

for comp, nodes in (
    component_node_lists.items()
):
    all_ancestors = set()

    for node in nodes:
        all_ancestors.update(
            ancestor_chain(node)
        )

    for anc in all_ancestors:
        amrf_node_to_components[
            anc
        ].add(comp)

node_component_count = {
    node: len(comps)
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


component_size_lookup = (
    component_stats.set_index(
        "homology_component"
    )
)

amrf_anchor_rows = []

for comp in sorted(
    positive_only_components
):
    total_n = int(
        component_size_lookup.loc[
            comp,
            "proteins",
        ]
    )

    mapped_n = int(
        protein_nodes.loc[
            protein_nodes[
                "homology_component"
            ] == comp,
            "protein_id",
        ].nunique()
    )

    lca = component_lca.get(
        comp
    )

    anchor = coverage_anchor(
        lca
    )

    amrf_anchor_rows.append({
        "homology_component":
            comp,

        "proteins":
            total_n,

        "amrf_mapped_proteins":
            mapped_n,

        "mapping_fraction":
            mapped_n / total_n,

        "component_lca":
            lca,

        "coverage_anchor":
            anchor,

        "anchor_descendant_components":
            (
                node_component_count.get(
                    anchor,
                    0,
                )
                if anchor is not None
                else 0
            ),
    })

amrf_component = pd.DataFrame(
    amrf_anchor_rows
)

assert (
    amrf_component[
        "coverage_anchor"
    ].notna().sum()
    == 185
)

print(
    "AMRFinder component anchors: PASS"
)


# ============================================================
# CARD BIOLOGICAL STRUCTURE
# ============================================================

card = json.loads(
    CARD_JSON_PATH.read_text(
        encoding="utf-8"
    )
)

assert str(
    card["_version"]
) == EXPECTED_CARD_VERSION

card_fasta_rows = []

with CARD_FASTA_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.startswith(">"):
            continue

        body = line[1:].strip()
        parts = body.split("|")

        aro_tokens = [
            x
            for x in parts
            if x.startswith("ARO:")
        ]

        assert len(
            aro_tokens
        ) == 1

        aro = (
            aro_tokens[0]
            .replace("ARO:", "")
            .strip()
        )

        if (
            len(parts) >= 2
            and parts[0]
            in {"gb", "ref", "sp"}
        ):
            acc = parts[1]
        else:
            acc = parts[0]

        card_fasta_rows.append({
            "accession_norm":
                norm_acc(acc),

            "ARO_accession":
                str(aro),
        })

card_fasta_index = (
    pd.DataFrame(
        card_fasta_rows
    )
    .drop_duplicates()
)

aro_rows = []

for key, entry in card.items():
    if not str(key).isdigit():
        continue

    aro = str(
        entry.get(
            "ARO_accession",
            "",
        )
    ).strip()

    if not aro:
        continue

    families = set()
    mechanisms = set()
    drugs = set()

    for cat in (
        entry.get(
            "ARO_category",
            {},
        )
        or {}
    ).values():

        cls = cat.get(
            "category_aro_class_name"
        )

        name = cat.get(
            "category_aro_name"
        )

        if not name:
            continue

        if cls == "AMR Gene Family":
            families.add(
                str(name)
            )
        elif cls == "Resistance Mechanism":
            mechanisms.add(
                str(name)
            )
        elif cls == "Drug Class":
            drugs.add(
                str(name)
            )

    aro_rows.append({
        "ARO_accession":
            aro,

        "ARO_name":
            entry.get("ARO_name"),

        "amr_gene_families":
            tuple(
                sorted(families)
            ),

        "resistance_mechanisms":
            tuple(
                sorted(mechanisms)
            ),

        "drug_classes":
            tuple(
                sorted(drugs)
            ),
    })

aro_meta = pd.DataFrame(
    aro_rows
)


def tuple_union(series):
    out = set()

    for value in series:
        if isinstance(
            value,
            (tuple, list, set),
        ):
            out.update(value)

    return tuple(
        sorted(out)
    )


card_supported = pos[
    pos["sources"]
    .astype(str)
    .str.contains(
        "CARD",
        regex=False,
    )
].copy()

card_acc_rows = []

for row in card_supported.itertuples(
    index=False
):
    raw = row.source_accessions

    if pd.isna(raw):
        continue

    for token in str(raw).split(";"):
        token = token.strip()

        if not token:
            continue

        card_acc_rows.append({
            "protein_id":
                row.protein_id,

            "accession_norm":
                norm_acc(token),
        })

all_card_map = (
    pd.DataFrame(
        card_acc_rows
    )
    .merge(
        card_fasta_index,
        on="accession_norm",
        how="left",
    )
)

mapped_card_ids = set(
    all_card_map.loc[
        all_card_map[
            "ARO_accession"
        ].notna(),
        "protein_id",
    ]
)

assert mapped_card_ids == set(
    card_supported["protein_id"]
)

all_card_ann = (
    all_card_map
    .dropna(
        subset=["ARO_accession"]
    )
    .merge(
        aro_meta,
        on="ARO_accession",
        how="left",
    )
)

assert all_card_ann[
    "ARO_name"
].notna().all()

all_card_protein = (
    all_card_ann
    .groupby("protein_id")
    .agg(
        card_gene_families=(
            "amr_gene_families",
            tuple_union,
        ),

        card_mechanisms=(
            "resistance_mechanisms",
            tuple_union,
        ),

        card_drug_classes=(
            "drug_classes",
            tuple_union,
        ),
    )
    .reset_index()
    .merge(
        components[
            [
                "protein_id",
                "homology_component",
            ]
        ],
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
)

assert len(
    all_card_protein
) == 5466

assert all_card_protein[
    "card_gene_families"
].map(len).gt(0).all()

card_family_to_components = defaultdict(
    set
)

component_to_card_families = defaultdict(
    set
)

for row in all_card_protein.itertuples(
    index=False
):
    for fam in row.card_gene_families:
        card_family_to_components[
            fam
        ].add(
            row.homology_component
        )

        component_to_card_families[
            row.homology_component
        ].add(fam)


# ============================================================
# CARD RESCUE FOR ZERO-AMRF COMPONENTS
# ============================================================

zero_amrf_components = set(
    amrf_component.loc[
        amrf_component[
            "amrf_mapped_proteins"
        ] == 0,
        "homology_component",
    ]
)

assert len(
    zero_amrf_components
) == 98

zero_card = all_card_protein[
    all_card_protein[
        "homology_component"
    ].isin(
        zero_amrf_components
    )
].copy()

card_common_rows = []

for comp, g in zero_card.groupby(
    "homology_component"
):
    fam_sets = [
        set(x)
        for x in g[
            "card_gene_families"
        ]
    ]

    mech_sets = [
        set(x)
        for x in g[
            "card_mechanisms"
        ]
    ]

    drug_sets = [
        set(x)
        for x in g[
            "card_drug_classes"
        ]
    ]

    card_common_rows.append({
        "homology_component":
            comp,

        "proteins":
            int(
                component_size_lookup.loc[
                    comp,
                    "proteins",
                ]
            ),

        "common_gene_families":
            tuple(
                sorted(
                    set.intersection(
                        *fam_sets
                    )
                    if fam_sets
                    else set()
                )
            ),

        "common_mechanisms":
            tuple(
                sorted(
                    set.intersection(
                        *mech_sets
                    )
                    if mech_sets
                    else set()
                )
            ),

        "common_drug_classes":
            tuple(
                sorted(
                    set.intersection(
                        *drug_sets
                    )
                    if drug_sets
                    else set()
                )
            ),
    })

card_common = pd.DataFrame(
    card_common_rows
)

assert len(
    card_common
) == 98

family_component_count = Counter()
mechanism_component_count = Counter()

for row in card_common.itertuples(
    index=False
):
    for fam in (
        row.common_gene_families
    ):
        family_component_count[
            fam
        ] += 1

    for mech in (
        row.common_mechanisms
    ):
        mechanism_component_count[
            mech
        ] += 1


def choose_card_anchor(row):
    fams = [
        x
        for x in row.common_gene_families
        if family_component_count[
            x
        ] >= 3
    ]

    if fams:
        chosen = sorted(
            fams,
            key=lambda x: (
                family_component_count[x],
                x,
            ),
        )[0]

        return (
            "CARD_FAMILY",
            chosen,
            family_component_count[
                chosen
            ],
        )

    mechs = [
        x
        for x in row.common_mechanisms
        if mechanism_component_count[
            x
        ] >= 3
    ]

    if mechs:
        chosen = sorted(
            mechs,
            key=lambda x: (
                mechanism_component_count[x],
                x,
            ),
        )[0]

        return (
            "CARD_MECHANISM",
            chosen,
            mechanism_component_count[
                chosen
            ],
        )

    return (None, None, 0)


chosen = card_common.apply(
    choose_card_anchor,
    axis=1,
    result_type="expand",
)

chosen.columns = [
    "card_anchor_type",
    "card_anchor",
    "card_anchor_component_count",
]

card_common = pd.concat(
    [
        card_common.reset_index(
            drop=True
        ),
        chosen.reset_index(
            drop=True
        ),
    ],
    axis=1,
)

assert (
    card_common[
        "card_anchor"
    ].notna().sum()
    == 96
)

print("CARD component anchors: PASS")


# ============================================================
# UNIFIED PRIMARY ANCHORS
# ============================================================

unified = amrf_component.copy()

unified[
    "primary_anchor_source"
] = np.where(
    unified[
        "coverage_anchor"
    ].notna(),
    "AMRFinderPlus",
    None,
)

unified[
    "primary_anchor_type"
] = np.where(
    unified[
        "coverage_anchor"
    ].notna(),
    "AMRF_HIERARCHY",
    None,
)

unified[
    "primary_anchor"
] = unified[
    "coverage_anchor"
]

unified[
    "primary_anchor_span_components"
] = unified[
    "anchor_descendant_components"
]

card_lookup = card_common.set_index(
    "homology_component"
)

for idx, row in unified.iterrows():
    comp = row[
        "homology_component"
    ]

    if (
        pd.isna(
            row[
                "primary_anchor"
            ]
        )
        and comp in card_lookup.index
    ):
        c = card_lookup.loc[
            comp
        ]

        if pd.notna(
            c["card_anchor"]
        ):
            unified.at[
                idx,
                "primary_anchor_source",
            ] = "CARD"

            unified.at[
                idx,
                "primary_anchor_type",
            ] = c[
                "card_anchor_type"
            ]

            unified.at[
                idx,
                "primary_anchor",
            ] = c[
                "card_anchor"
            ]

            unified.at[
                idx,
                "primary_anchor_span_components",
            ] = int(
                c[
                    "card_anchor_component_count"
                ]
            )

assert (
    unified[
        "primary_anchor"
    ].notna().sum()
    == 281
)

assert int(
    unified.loc[
        unified[
            "primary_anchor"
        ].notna(),
        "proteins",
    ].sum()
) == 5257

unified[
    "anchor_key"
] = np.where(
    unified[
        "primary_anchor"
    ].notna(),
    (
        unified[
            "primary_anchor_source"
        ].astype(str)
        + "::"
        + unified[
            "primary_anchor_type"
        ].astype(str)
        + "::"
        + unified[
            "primary_anchor"
        ].astype(str)
    ),
    None,
)

print("Unified primary anchors: PASS")


# ============================================================
# TRANSITIVE CROSS-SOURCE OOD GRAPH
# ============================================================

positive_component_universe = set(
    unified[
        "homology_component"
    ]
)

anchored_components = set(
    unified.loc[
        unified[
            "primary_anchor"
        ].notna(),
        "homology_component",
    ]
)

unanchored_components = (
    positive_component_universe
    - anchored_components
)

adj = {
    comp: set()
    for comp
    in positive_component_universe
}


def connect(a, b):
    if a == b:
        return

    if (
        a not in adj
        or b not in adj
    ):
        return

    adj[a].add(b)
    adj[b].add(a)


for row in unified.itertuples(
    index=False
):
    lca = row.component_lca

    if pd.isna(lca):
        continue

    if lca in ROOT_NODES:
        continue

    if row.mapping_fraction < 0.80:
        continue

    subtree = (
        set(
            amrf_node_to_components.get(
                lca,
                set(),
            )
        )
        & positive_component_universe
    )

    for other in subtree:
        connect(
            row.homology_component,
            other,
        )


for fam, comps in (
    card_family_to_components.items()
):
    comps = sorted(
        set(comps)
        & positive_component_universe
    )

    for i in range(
        len(comps)
    ):
        for j in range(
            i + 1,
            len(comps),
        ):
            connect(
                comps[i],
                comps[j],
            )


amrf_seed_components = set(
    unified.loc[
        unified[
            "primary_anchor"
        ].isna()
        & unified[
            "component_lca"
        ].notna()
        & ~unified[
            "component_lca"
        ].isin(
            ROOT_NODES
        )
        & (
            unified[
                "mapping_fraction"
            ] >= 0.80
        ),
        "homology_component",
    ]
)

card_seed_components = set()

for row in card_common.itertuples(
    index=False
):
    if pd.notna(
        row.card_anchor
    ):
        continue

    if len(
        row.common_gene_families
    ) == 0:
        continue

    card_seed_components.add(
        row.homology_component
    )

seed_components = (
    amrf_seed_components
    | card_seed_components
)


def graph_closure(start):
    seen = {start}
    q = deque([start])

    while q:
        current = q.popleft()

        for nxt in adj[
            current
        ]:
            if nxt in seen:
                continue

            seen.add(nxt)
            q.append(nxt)

    return seen


candidate_blocks = []
seen_seeds = set()

for seed in sorted(
    seed_components
):
    if seed in seen_seeds:
        continue

    closure = graph_closure(
        seed
    )

    seeds_inside = (
        closure
        & seed_components
    )

    seen_seeds.update(
        seeds_inside
    )

    candidate_blocks.append({
        "seeds":
            tuple(
                sorted(
                    seeds_inside
                )
            ),

        "closure":
            tuple(
                sorted(
                    closure
                )
            ),
    })

eligible_blocks = []

for block in candidate_blocks:
    closure = set(
        block["closure"]
    )

    if (
        closure
        & anchored_components
    ):
        continue

    eligible_blocks.append(
        closure
    )

final_ood_components = set()

for block in eligible_blocks:
    final_ood_components.update(
        block
    )

assert len(
    eligible_blocks
) == 24

assert len(
    final_ood_components
) == 29

assert final_ood_components.isdisjoint(
    anchored_components
)

ood_positive_count = int(
    component_stats.loc[
        component_stats[
            "homology_component"
        ].isin(
            final_ood_components
        ),
        "positives",
    ].sum()
)

assert (
    ood_positive_count
    == OOD_POS_EXPECTED
)

ood_block_lookup = {}

eligible_blocks_sorted = sorted(
    eligible_blocks,
    key=lambda x: sorted(x)[0],
)

for block_i, block in enumerate(
    eligible_blocks_sorted,
    start=1,
):
    block_id = (
        f"OOD_BLOCK_{block_i:03d}"
    )

    for comp in block:
        ood_block_lookup[
            comp
        ] = block_id

print(
    "Transitive OOD closure: PASS"
)


# ============================================================
# PROTEIN-LEVEL BIOLOGICAL CATEGORY METADATA
# ============================================================

protein_category_sets = defaultdict(
    set
)

for row in mapped_amrf.itertuples(
    index=False
):
    if pd.notna(
        row.amrf_class
    ):
        protein_category_sets[
            row.protein_id
        ].add(
            "AMRF_CLASS::"
            + str(
                row.amrf_class
            )
        )

    if pd.notna(
        row.amrf_subclass
    ):
        protein_category_sets[
            row.protein_id
        ].add(
            "AMRF_SUBCLASS::"
            + str(
                row.amrf_subclass
            )
        )


for row in all_card_protein.itertuples(
    index=False
):
    for mech in (
        row.card_mechanisms
    ):
        protein_category_sets[
            row.protein_id
        ].add(
            "CARD_MECHANISM::"
            + mech
        )

    for drug in (
        row.card_drug_classes
    ):
        protein_category_sets[
            row.protein_id
        ].add(
            "CARD_DRUG::"
            + drug
        )


# ============================================================
# POSITIVE PRIMARY SPLIT OPTIMIZATION
# ============================================================

positive_primary = pos.merge(
    components[
        [
            "protein_id",
            "homology_component",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
)

primary_candidate_components = set(
    unified.loc[
        unified[
            "primary_anchor"
        ].notna()
        & ~unified[
            "homology_component"
        ].isin(
            final_ood_components
        )
        & (
            unified[
                "proteins"
            ] <= PRIMARY_COMPONENT_CAP
        ),
        "homology_component",
    ]
)

candidate_component_rows = (
    unified[
        unified[
            "homology_component"
        ].isin(
            primary_candidate_components
        )
    ]
    .copy()
    .set_index(
        "homology_component"
    )
)

candidate_proteins = (
    positive_primary[
        positive_primary[
            "homology_component"
        ].isin(
            primary_candidate_components
        )
    ]
    .copy()
)

candidate_pool_n = len(
    candidate_proteins
)

candidate_ids_by_component = (
    candidate_proteins.groupby(
        "homology_component"
    )["protein_id"]
    .apply(list)
    .to_dict()
)

candidate_lengths_by_component = (
    candidate_proteins.groupby(
        "homology_component"
    )["length"]
    .apply(
        lambda x:
            list(
                map(int, x)
            )
    )
    .to_dict()
)

candidate_sources_by_component = {}

for comp, g in (
    candidate_proteins.groupby(
        "homology_component"
    )
):
    candidate_sources_by_component[
        comp
    ] = Counter(
        g["sources"]
        .astype(str)
    )

component_category_counts = {}
category_protein_count = Counter()
category_component_count = Counter()

for comp, ids in (
    candidate_ids_by_component.items()
):
    counts = Counter()
    present = set()

    for protein_id in ids:
        for token in (
            protein_category_sets.get(
                protein_id,
                set(),
            )
        ):
            counts[token] += 1
            present.add(token)

    component_category_counts[
        comp
    ] = counts

    category_protein_count.update(
        counts
    )

    for token in present:
        category_component_count[
            token
        ] += 1


common_category_min_proteins = math.ceil(
    float(
        COMMON_CATEGORY_MIN_FRACTION
    )
    * candidate_pool_n
)

common_categories = sorted(
    token
    for token, count
    in category_protein_count.items()
    if (
        count
        >= common_category_min_proteins
        and category_component_count[
            token
        ]
        >= COMMON_CATEGORY_MIN_COMPONENTS
    )
)

print(
    "Positive evaluation candidate proteins:",
    candidate_pool_n
)

print(
    "Common resistance-category tokens (audit only):",
    len(common_categories)
)

model = cp_model.CpModel()

candidate_components_sorted = sorted(
    primary_candidate_components
)

x_val = {
    comp:
        model.NewBoolVar(
            f"VAL__{comp}"
        )
    for comp in candidate_components_sorted
}

x_test = {
    comp:
        model.NewBoolVar(
            f"TEST__{comp}"
        )
    for comp in candidate_components_sorted
}

for comp in candidate_components_sorted:
    model.Add(
        x_val[comp]
        + x_test[comp]
        <= 1
    )

model.Add(
    sum(
        int(
            candidate_component_rows.loc[
                comp,
                "proteins",
            ]
        )
        * x_val[comp]
        for comp
        in candidate_components_sorted
    )
    == VAL_POS_TARGET
)

model.Add(
    sum(
        int(
            candidate_component_rows.loc[
                comp,
                "proteins",
            ]
        )
        * x_test[comp]
        for comp
        in candidate_components_sorted
    )
    == TEST_POS_TARGET
)


# ------------------------------------------------------------
# Every evaluation anchor must retain a training component.
# ------------------------------------------------------------

anchored_nonood = unified[
    unified[
        "primary_anchor"
    ].notna()
    & ~unified[
        "homology_component"
    ].isin(
        final_ood_components
    )
].copy()

for anchor_key, g in (
    anchored_nonood.groupby(
        "anchor_key"
    )
):
    all_comps = set(
        g[
            "homology_component"
        ]
    )

    variable_comps = sorted(
        all_comps
        & primary_candidate_components
    )

    fixed_train = (
        all_comps
        - primary_candidate_components
    )

    if fixed_train:
        continue

    if variable_comps:
        model.Add(
            sum(
                x_val[c]
                + x_test[c]
                for c in variable_comps
            )
            <= len(
                variable_comps
            ) - 1
        )


# ------------------------------------------------------------
# Hard primary-positive length constraints.
#
# Validation and test must resemble the BIOLOGICALLY ELIGIBLE
# positive evaluation pool, not the complete training set.
#
# Required:
#   validation vs eligible pool KS <= 0.10
#   test vs eligible pool KS <= 0.10
#   validation vs test KS <= 0.10
#
# Positive train-vs-evaluation KS is report-only because most
# training positives are structurally ineligible for ordinary
# evaluation under the frozen biological design.
# ------------------------------------------------------------

eligible_positive_thresholds = sorted(
    set(
        map(
            int,
            candidate_proteins[
                "length"
            ],
        )
    )
)

ks_bound_pool = math.floor(
    float(KS_LIMIT)
    * candidate_pool_n
    * VAL_POS_TARGET
)

ks_bound_val_test = math.floor(
    float(KS_LIMIT)
    * VAL_POS_TARGET
    * TEST_POS_TARGET
)

for threshold in eligible_positive_thresholds:

    pool_cum = int(
        (
            candidate_proteins[
                "length"
            ] <= threshold
        ).sum()
    )

    val_cum = sum(
        sum(
            length <= threshold
            for length
            in candidate_lengths_by_component[
                comp
            ]
        )
        * x_val[comp]
        for comp
        in candidate_components_sorted
    )

    test_cum = sum(
        sum(
            length <= threshold
            for length
            in candidate_lengths_by_component[
                comp
            ]
        )
        * x_test[comp]
        for comp
        in candidate_components_sorted
    )

    expr_val_pool = (
        candidate_pool_n
        * val_cum
        - VAL_POS_TARGET
        * pool_cum
    )

    expr_test_pool = (
        candidate_pool_n
        * test_cum
        - TEST_POS_TARGET
        * pool_cum
    )

    expr_val_test = (
        TEST_POS_TARGET
        * val_cum
        - VAL_POS_TARGET
        * test_cum
    )

    model.Add(
        expr_val_pool
        <= ks_bound_pool
    )
    model.Add(
        expr_val_pool
        >= -ks_bound_pool
    )

    model.Add(
        expr_test_pool
        <= ks_bound_pool
    )
    model.Add(
        expr_test_pool
        >= -ks_bound_pool
    )

    model.Add(
        expr_val_test
        <= ks_bound_val_test
    )
    model.Add(
        expr_val_test
        >= -ks_bound_val_test
    )


# ------------------------------------------------------------
# Hard source-provenance tolerance: +/-5 percentage points.
# ------------------------------------------------------------

candidate_source_totals = Counter(
    candidate_proteins[
        "sources"
    ].astype(str)
)

for source_name, pool_count in (
    sorted(
        candidate_source_totals.items()
    )
):
    lower, upper = fraction_bounds(
        pool_count,
        candidate_pool_n,
        VAL_POS_TARGET,
        Fraction(5, 100),
    )

    val_source = sum(
        candidate_sources_by_component[
            comp
        ].get(
            source_name,
            0,
        )
        * x_val[comp]
        for comp
        in candidate_components_sorted
    )

    test_source = sum(
        candidate_sources_by_component[
            comp
        ].get(
            source_name,
            0,
        )
        * x_test[comp]
        for comp
        in candidate_components_sorted
    )

    model.Add(
        val_source >= lower
    )
    model.Add(
        val_source <= upper
    )

    model.Add(
        test_source >= lower
    )
    model.Add(
        test_source <= upper
    )


# ------------------------------------------------------------
# Stage 1: length-bin proportionality.
# ------------------------------------------------------------

length_devs = []

for lo, hi in LENGTH_BINS:
    pool_bin_count = int(
        (
            candidate_proteins[
                "length"
            ].between(
                lo,
                hi,
                inclusive="both",
            )
        ).sum()
    )

    val_bin = sum(
        sum(
            lo <= length <= hi
            for length
            in candidate_lengths_by_component[
                comp
            ]
        )
        * x_val[comp]
        for comp
        in candidate_components_sorted
    )

    test_bin = sum(
        sum(
            lo <= length <= hi
            for length
            in candidate_lengths_by_component[
                comp
            ]
        )
        * x_test[comp]
        for comp
        in candidate_components_sorted
    )

    for name, expr in [
        ("V", val_bin),
        ("T", test_bin),
    ]:
        dev = model.NewIntVar(
            0,
            candidate_pool_n
            * VAL_POS_TARGET,
            f"LENDEV_{lo}_{hi}_{name}",
        )

        model.AddAbsEquality(
            dev,
            candidate_pool_n
            * expr
            - pool_bin_count
            * VAL_POS_TARGET,
        )

        length_devs.append(
            dev
        )

length_objective = sum(
    length_devs
)


# ------------------------------------------------------------
# Stage 2: source proportionality.
# ------------------------------------------------------------

source_devs = []

for source_name, pool_count in (
    sorted(
        candidate_source_totals.items()
    )
):
    val_source = sum(
        candidate_sources_by_component[
            comp
        ].get(
            source_name,
            0,
        )
        * x_val[comp]
        for comp
        in candidate_components_sorted
    )

    test_source = sum(
        candidate_sources_by_component[
            comp
        ].get(
            source_name,
            0,
        )
        * x_test[comp]
        for comp
        in candidate_components_sorted
    )

    safe_name = hashlib.sha256(
        source_name.encode()
    ).hexdigest()[:8]

    for name, expr in [
        ("V", val_source),
        ("T", test_source),
    ]:
        dev = model.NewIntVar(
            0,
            candidate_pool_n
            * VAL_POS_TARGET,
            f"SRCDEV_{safe_name}_{name}",
        )

        model.AddAbsEquality(
            dev,
            candidate_pool_n
            * expr
            - pool_count
            * VAL_POS_TARGET,
        )

        source_devs.append(
            dev
        )

source_objective = sum(
    source_devs
)


# ------------------------------------------------------------
# Resistance-category metadata for post-split audit.
# ------------------------------------------------------------

category_devs = []

for token in common_categories:
    pool_count = int(
        category_protein_count[
            token
        ]
    )

    val_count = sum(
        component_category_counts[
            comp
        ].get(
            token,
            0,
        )
        * x_val[comp]
        for comp
        in candidate_components_sorted
    )

    test_count = sum(
        component_category_counts[
            comp
        ].get(
            token,
            0,
        )
        * x_test[comp]
        for comp
        in candidate_components_sorted
    )

    token_name = hashlib.sha256(
        token.encode()
    ).hexdigest()[:10]

    for name, expr in [
        ("V", val_count),
        ("T", test_count),
    ]:
        dev = model.NewIntVar(
            0,
            candidate_pool_n
            * VAL_POS_TARGET,
            f"CATDEV_{token_name}_{name}",
        )

        model.AddAbsEquality(
            dev,
            candidate_pool_n
            * expr
            - pool_count
            * VAL_POS_TARGET,
        )

        category_devs.append(
            dev
        )

category_objective = (
    sum(category_devs)
    if category_devs
    else 0
)


# ------------------------------------------------------------
# Stage 3: biological-anchor representation/symmetry.
# ------------------------------------------------------------

anchor_both = []
anchor_any = []

candidate_anchor_groups = (
    candidate_component_rows
    .reset_index()
    .groupby(
        "anchor_key"
    )[
        "homology_component"
    ]
    .apply(list)
)

for anchor_i, (
    anchor_key,
    comps,
) in enumerate(
    candidate_anchor_groups.items()
):
    val_present = model.NewBoolVar(
        f"ANCHOR_VAL_{anchor_i}"
    )

    test_present = model.NewBoolVar(
        f"ANCHOR_TEST_{anchor_i}"
    )

    both = model.NewBoolVar(
        f"ANCHOR_BOTH_{anchor_i}"
    )

    val_sum = sum(
        x_val[c]
        for c in comps
    )

    test_sum = sum(
        x_test[c]
        for c in comps
    )

    model.Add(
        val_sum >= 1
    ).OnlyEnforceIf(
        val_present
    )

    model.Add(
        val_sum == 0
    ).OnlyEnforceIf(
        val_present.Not()
    )

    model.Add(
        test_sum >= 1
    ).OnlyEnforceIf(
        test_present
    )

    model.Add(
        test_sum == 0
    ).OnlyEnforceIf(
        test_present.Not()
    )

    model.Add(
        both <= val_present
    )
    model.Add(
        both <= test_present
    )
    model.Add(
        both
        >= val_present
        + test_present
        - 1
    )

    anchor_both.append(
        both
    )

    anchor_any.extend(
        [
            val_present,
            test_present,
        ]
    )

anchor_objective = (
    1000
    * sum(anchor_both)
    + sum(anchor_any)
)


# ------------------------------------------------------------
# Stage 4: deterministic SHA-256 tie break.
# ------------------------------------------------------------

val_ranks = deterministic_rank(
    candidate_components_sorted,
    "POSITIVE_VAL",
)

test_ranks = deterministic_rank(
    candidate_components_sorted,
    "POSITIVE_TEST",
)

tie_objective = sum(
    val_ranks[c]
    * x_val[c]
    + test_ranks[c]
    * x_test[c]
    for c in candidate_components_sorted
)


# ------------------------------------------------------------
# Lexicographic solve.
# ------------------------------------------------------------

positive_objectives = {}

positive_objectives[
    "length"
] = solve_optimal(
    model,
    length_objective,
    "min",
    "Positive stage 1 / length",
)

positive_objectives[
    "source"
] = solve_optimal(
    model,
    source_objective,
    "min",
    "Positive stage 2 / source",
)

positive_objectives[
    "categories"
] = {
    "mode": "audit_only",
    "common_tokens": len(common_categories),
}

positive_objectives[
    "anchors"
] = solve_optimal(
    model,
    anchor_objective,
    "max",
    "Positive stage 3 / anchors",
)

# ------------------------------------------------------------
# Final positive selection.
#
# The three biologically meaningful objectives above have
# already reached global OPTIMAL status and solve_optimal()
# has fixed each optimum back into the model as a hard
# equality. No arbitrary hash-ranking objective is needed.
#
# Select one feasible assignment satisfying the complete
# frozen constraint set and all three fixed optima.
# ------------------------------------------------------------

model.Minimize(0)

positive_solver = cp_model.CpSolver()
positive_solver.parameters.num_search_workers = 1
positive_solver.parameters.random_seed = SPLIT_SEED
positive_solver.parameters.max_time_in_seconds = 300
positive_solver.parameters.stop_after_first_solution = True

positive_status = positive_solver.Solve(
    model
)

if positive_status not in (
    cp_model.FEASIBLE,
    cp_model.OPTIMAL,
):
    raise RuntimeError(
        "Positive final feasibility selection failed. "
        f"Status={positive_solver.StatusName(positive_status)}"
    )

positive_objectives[
    "tie_break"
] = 0

positive_objectives[
    "tie_break_mode"
] = "not_optimized"

positive_objectives[
    "selection_mode"
] = (
    "fixed_seed_single_worker_feasible_after_"
    "length_source_anchor_optima"
)

positive_objectives[
    "selection_status"
] = positive_solver.StatusName(
    positive_status
)

print(
    "Positive final feasibility selection:",
    positive_solver.StatusName(
        positive_status
    ),
)


positive_component_partition = {}

for comp in positive_component_universe:
    if comp in final_ood_components:
        positive_component_partition[
            comp
        ] = "ood"

    elif (
        comp
        in primary_candidate_components
        and positive_solver.Value(
            x_val[comp]
        )
        == 1
    ):
        positive_component_partition[
            comp
        ] = "validation"

    elif (
        comp
        in primary_candidate_components
        and positive_solver.Value(
            x_test[comp]
        )
        == 1
    ):
        positive_component_partition[
            comp
        ] = "test"

    else:
        positive_component_partition[
            comp
        ] = "train"


# Mixed components are always train.
for comp in mixed_components:
    positive_component_partition[
        comp
    ] = "train"


positive_partition_by_id = {}

for row in pos.itertuples(
    index=False
):
    comp = components.loc[
        components[
            "protein_id"
        ] == row.protein_id,
        "homology_component",
    ].iloc[0]

    positive_partition_by_id[
        row.protein_id
    ] = positive_component_partition[
        comp
    ]

positive_counts = Counter(
    positive_partition_by_id.values()
)

assert positive_counts == {
    "train": TRAIN_POS_EXPECTED,
    "validation": VAL_POS_TARGET,
    "test": TEST_POS_TARGET,
    "ood": OOD_POS_EXPECTED,
}

print(
    "Positive partition counts: PASS",
    dict(positive_counts),
)


# ============================================================
# NEGATIVE SPLIT OPTIMIZATION
# ============================================================

negative_basic = neg.merge(
    components[
        [
            "protein_id",
            "homology_component",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
)

negative_only_df = negative_basic[
    negative_basic[
        "homology_component"
    ].isin(
        negative_only_components
    )
].copy()

neg_ids_by_component = (
    negative_only_df.groupby(
        "homology_component"
    )["protein_id"]
    .apply(list)
    .to_dict()
)

neg_lengths_by_component = (
    negative_only_df.groupby(
        "homology_component"
    )["length"]
    .apply(
        lambda x:
            list(
                map(int, x)
            )
    )
    .to_dict()
)

neg_components_sorted = sorted(
    negative_only_components
)

neg_size = {
    comp:
        len(
            neg_ids_by_component[
                comp
            ]
        )
    for comp in neg_components_sorted
}

neg_model = cp_model.CpModel()

n_val = {
    comp:
        neg_model.NewBoolVar(
            f"NVAL__{comp}"
        )
    for comp in neg_components_sorted
}

n_test = {
    comp:
        neg_model.NewBoolVar(
            f"NTEST__{comp}"
        )
    for comp in neg_components_sorted
}

n_ood = {
    comp:
        neg_model.NewBoolVar(
            f"NOOD__{comp}"
        )
    for comp in neg_components_sorted
}

for comp in neg_components_sorted:
    neg_model.Add(
        n_val[comp]
        + n_test[comp]
        + n_ood[comp]
        <= 1
    )

    if (
        neg_size[comp]
        > PRIMARY_COMPONENT_CAP
    ):
        neg_model.Add(
            n_val[comp] == 0
        )
        neg_model.Add(
            n_test[comp] == 0
        )

    if (
        neg_size[comp]
        > OOD_NEG_COMPONENT_CAP
    ):
        neg_model.Add(
            n_ood[comp] == 0
        )


neg_model.Add(
    sum(
        neg_size[c]
        * n_val[c]
        for c in neg_components_sorted
    )
    == VAL_NEG_TARGET
)

neg_model.Add(
    sum(
        neg_size[c]
        * n_test[c]
        for c in neg_components_sorted
    )
    == TEST_NEG_TARGET
)

neg_model.Add(
    sum(
        neg_size[c]
        * n_ood[c]
        for c in neg_components_sorted
    )
    == OOD_NEG_TARGET
)


# ------------------------------------------------------------
# Positive target length arrays.
# ------------------------------------------------------------

pos_with_partition = pos[
    [
        "protein_id",
        "length",
    ]
].copy()

pos_with_partition[
    "partition"
] = pos_with_partition[
    "protein_id"
].map(
    positive_partition_by_id
)

pos_lengths = {
    partition:
        pos_with_partition.loc[
            pos_with_partition[
                "partition"
            ] == partition,
            "length",
        ]
        .astype(int)
        .to_numpy()
    for partition in [
        "train",
        "validation",
        "test",
        "ood",
    ]
}


# ------------------------------------------------------------
# Exact within-partition positive-vs-negative KS constraints.
#
# Required for ALL four partitions:
#   train      positive vs negative KS <= 0.10
#   validation positive vs negative KS <= 0.10
#   test       positive vs negative KS <= 0.10
#   OOD        positive vs negative KS <= 0.10
#
# The 35 negative proteins in mixed components are fixed in
# training and are included in the training negative CDF.
# ------------------------------------------------------------

all_neg_lengths = (
    neg["length"]
    .astype(int)
    .to_numpy()
)

all_thresholds = sorted(
    set(
        np.concatenate(
            [
                all_neg_lengths,
                pos_lengths[
                    "train"
                ],
                pos_lengths[
                    "validation"
                ],
                pos_lengths[
                    "test"
                ],
                pos_lengths[
                    "ood"
                ],
            ]
        )
    )
)

partition_targets = {
    "train":
        TRAIN_NEG_EXPECTED,

    "validation":
        VAL_NEG_TARGET,

    "test":
        TEST_NEG_TARGET,

    "ood":
        OOD_NEG_TARGET,
}

ks_count_bounds = {
    partition:
        math.floor(
            float(KS_LIMIT)
            * n
        )
    for partition, n
    in partition_targets.items()
}

for threshold in all_thresholds:

    val_neg_cum = sum(
        sum(
            length <= threshold
            for length
            in neg_lengths_by_component[
                comp
            ]
        )
        * n_val[comp]
        for comp
        in neg_components_sorted
    )

    test_neg_cum = sum(
        sum(
            length <= threshold
            for length
            in neg_lengths_by_component[
                comp
            ]
        )
        * n_test[comp]
        for comp
        in neg_components_sorted
    )

    ood_neg_cum = sum(
        sum(
            length <= threshold
            for length
            in neg_lengths_by_component[
                comp
            ]
        )
        * n_ood[comp]
        for comp
        in neg_components_sorted
    )

    total_neg_cum = int(
        (
            neg["length"]
            .astype(int)
            <= threshold
        ).sum()
    )

    train_neg_cum = (
        total_neg_cum
        - val_neg_cum
        - test_neg_cum
        - ood_neg_cum
    )

    neg_cum = {
        "train":
            train_neg_cum,

        "validation":
            val_neg_cum,

        "test":
            test_neg_cum,

        "ood":
            ood_neg_cum,
    }

    for partition in [
        "train",
        "validation",
        "test",
        "ood",
    ]:

        pos_cum = int(
            (
                pos_lengths[
                    partition
                ]
                <= threshold
            ).sum()
        )

        bound = ks_count_bounds[
            partition
        ]

        neg_model.Add(
            neg_cum[
                partition
            ]
            - pos_cum
            <= bound
        )

        neg_model.Add(
            neg_cum[
                partition
            ]
            - pos_cum
            >= -bound
        )


# ------------------------------------------------------------
# Negative optimization objective:
# after satisfying exact empirical KS constraints, minimize
# six-bin positive-vs-negative deviations across all four
# partitions.
#
# This is deliberately much smaller than optimizing all 420
# empirical-CDF thresholds and keeps the exact KS rule hard.
# ------------------------------------------------------------

negative_bin_devs = []

for lo, hi in LENGTH_BINS:

    val_neg_bin = sum(
        sum(
            lo <= length <= hi
            for length
            in neg_lengths_by_component[
                comp
            ]
        )
        * n_val[comp]
        for comp
        in neg_components_sorted
    )

    test_neg_bin = sum(
        sum(
            lo <= length <= hi
            for length
            in neg_lengths_by_component[
                comp
            ]
        )
        * n_test[comp]
        for comp
        in neg_components_sorted
    )

    ood_neg_bin = sum(
        sum(
            lo <= length <= hi
            for length
            in neg_lengths_by_component[
                comp
            ]
        )
        * n_ood[comp]
        for comp
        in neg_components_sorted
    )

    total_neg_bin = int(
        neg["length"]
        .between(
            lo,
            hi,
            inclusive="both",
        )
        .sum()
    )

    train_neg_bin = (
        total_neg_bin
        - val_neg_bin
        - test_neg_bin
        - ood_neg_bin
    )

    neg_bin_expr = {
        "train":
            train_neg_bin,

        "validation":
            val_neg_bin,

        "test":
            test_neg_bin,

        "ood":
            ood_neg_bin,
    }

    for partition in [
        "train",
        "validation",
        "test",
        "ood",
    ]:

        pos_bin_count = int(
            (
                (
                    pos_lengths[
                        partition
                    ] >= lo
                )
                & (
                    pos_lengths[
                        partition
                    ] <= hi
                )
            ).sum()
        )

        dev = neg_model.NewIntVar(
            0,
            partition_targets[
                partition
            ],
            (
                f"NEG_BIN_DEV_"
                f"{partition}_"
                f"{lo}_{hi}"
            ),
        )

        neg_model.AddAbsEquality(
            dev,
            neg_bin_expr[
                partition
            ]
            - pos_bin_count,
        )

        negative_bin_devs.append(
            dev
        )


negative_length_objective = sum(
    negative_bin_devs
)

negative_tie_val = deterministic_rank(
    neg_components_sorted,
    "NEGATIVE_VAL",
)

negative_tie_test = deterministic_rank(
    neg_components_sorted,
    "NEGATIVE_TEST",
)

negative_tie_ood = deterministic_rank(
    neg_components_sorted,
    "NEGATIVE_OOD",
)

negative_tie_objective = sum(
    negative_tie_val[c]
    * n_val[c]
    + negative_tie_test[c]
    * n_test[c]
    + negative_tie_ood[c]
    * n_ood[c]
    for c in neg_components_sorted
)

# ------------------------------------------------------------
# Final negative selection.
#
# Exact counts, component integrity/caps, and all four
# within-partition positive-vs-negative empirical KS <= 0.10
# requirements are hard constraints.
#
# Six-bin deviations and SHA ranks are retained only as
# diagnostics / deterministic metadata; they are not global
# scientific optimization targets.
# ------------------------------------------------------------

negative_objectives = {
    "bin_length": 0,
    "bin_length_mode": "audit_only",
    "tie_break": 0,
    "tie_break_mode": "not_optimized",
    "selection_mode": (
        "fixed_seed_single_worker_feasible_under_"
        "hard_negative_constraints"
    ),
}

neg_model.Minimize(0)

negative_solver = cp_model.CpSolver()
negative_solver.parameters.num_search_workers = 1
negative_solver.parameters.random_seed = SPLIT_SEED
negative_solver.parameters.max_time_in_seconds = 300
negative_solver.parameters.stop_after_first_solution = True

negative_status = negative_solver.Solve(
    neg_model
)

if negative_status not in (
    cp_model.FEASIBLE,
    cp_model.OPTIMAL,
):
    raise RuntimeError(
        "Negative final feasibility selection failed. "
        f"Status={negative_solver.StatusName(negative_status)}"
    )

negative_objectives[
    "selection_status"
] = negative_solver.StatusName(
    negative_status
)

print(
    "Negative final feasibility selection:",
    negative_solver.StatusName(
        negative_status
    ),
)


negative_component_partition = {}

for comp in neg_components_sorted:
    if negative_solver.Value(
        n_val[comp]
    ):
        negative_component_partition[
            comp
        ] = "validation"

    elif negative_solver.Value(
        n_test[comp]
    ):
        negative_component_partition[
            comp
        ] = "test"

    elif negative_solver.Value(
        n_ood[comp]
    ):
        negative_component_partition[
            comp
        ] = "ood"

    else:
        negative_component_partition[
            comp
        ] = "train"


for comp in mixed_components:
    negative_component_partition[
        comp
    ] = "train"


negative_partition_by_id = {}

component_by_id = dict(
    zip(
        components[
            "protein_id"
        ],
        components[
            "homology_component"
        ],
    )
)

for row in neg.itertuples(
    index=False
):
    comp = component_by_id[
        row.protein_id
    ]

    negative_partition_by_id[
        row.protein_id
    ] = negative_component_partition[
        comp
    ]

negative_counts = Counter(
    negative_partition_by_id.values()
)

assert negative_counts == {
    "train": TRAIN_NEG_EXPECTED,
    "validation": VAL_NEG_TARGET,
    "test": TEST_NEG_TARGET,
    "ood": OOD_NEG_TARGET,
}

print(
    "Negative partition counts: PASS",
    dict(negative_counts),
)


# ============================================================
# COMBINE ASSIGNMENT
# ============================================================

partition_by_id = {}

partition_by_id.update(
    positive_partition_by_id
)

partition_by_id.update(
    negative_partition_by_id
)

assert len(
    partition_by_id
) == 15794

assignment = (
    corpus_basic[
        [
            "protein_id",
            "label",
            "length",
            "old_cdhit_cluster",
            "homology_component",
        ]
    ]
    .copy()
)

assignment[
    "partition"
] = assignment[
    "protein_id"
].map(
    partition_by_id
)

assert assignment[
    "partition"
].notna().all()

assert assignment[
    "protein_id"
].is_unique


# Biological metadata.
positive_group_lookup = unified.set_index(
    "homology_component"
)

assignment[
    "primary_anchor_source"
] = None

assignment[
    "primary_anchor_type"
] = None

assignment[
    "primary_anchor"
] = None

assignment[
    "component_lca"
] = None

assignment[
    "ood_block_id"
] = None

for idx, row in assignment[
    assignment[
        "label"
    ] == 1
].iterrows():
    comp = row[
        "homology_component"
    ]

    if comp in positive_group_lookup.index:
        u = positive_group_lookup.loc[
            comp
        ]

        assignment.at[
            idx,
            "primary_anchor_source",
        ] = u[
            "primary_anchor_source"
        ]

        assignment.at[
            idx,
            "primary_anchor_type",
        ] = u[
            "primary_anchor_type"
        ]

        assignment.at[
            idx,
            "primary_anchor",
        ] = u[
            "primary_anchor"
        ]

        assignment.at[
            idx,
            "component_lca",
        ] = u[
            "component_lca"
        ]

    assignment.at[
        idx,
        "ood_block_id",
    ] = ood_block_lookup.get(
        comp
    )


# ============================================================
# BUILDER SELF-CHECKS
# ============================================================

cross = pd.crosstab(
    assignment["partition"],
    assignment["label"],
)

expected_counts = {
    "train": {
        0: TRAIN_NEG_EXPECTED,
        1: TRAIN_POS_EXPECTED,
    },
    "validation": {
        0: VAL_NEG_TARGET,
        1: VAL_POS_TARGET,
    },
    "test": {
        0: TEST_NEG_TARGET,
        1: TEST_POS_TARGET,
    },
    "ood": {
        0: OOD_NEG_TARGET,
        1: OOD_POS_EXPECTED,
    },
}

for partition, counts in (
    expected_counts.items()
):
    for label, expected in (
        counts.items()
    ):
        assert int(
            cross.loc[
                partition,
                label,
            ]
        ) == expected


component_partition_counts = (
    assignment.groupby(
        "homology_component"
    )["partition"]
    .nunique()
)

assert (
    component_partition_counts == 1
).all()

mixed_outside_train = assignment[
    assignment[
        "homology_component"
    ].isin(
        mixed_components
    )
    & (
        assignment[
            "partition"
        ] != "train"
    )
]

assert len(
    mixed_outside_train
) == 0


for partition in [
    "validation",
    "test",
]:
    positive_eval = assignment[
        (
            assignment[
                "partition"
            ] == partition
        )
        & (
            assignment[
                "label"
            ] == 1
        )
    ]

    assert positive_eval[
        "primary_anchor"
    ].notna().all()

    max_size = int(
        positive_eval.groupby(
            "homology_component"
        ).size().max()
    )

    assert (
        max_size
        <= PRIMARY_COMPONENT_CAP
    )


for partition in [
    "validation",
    "test",
]:
    negative_eval = assignment[
        (
            assignment[
                "partition"
            ] == partition
        )
        & (
            assignment[
                "label"
            ] == 0
        )
    ]

    max_size = int(
        negative_eval.groupby(
            "homology_component"
        ).size().max()
    )

    assert (
        max_size
        <= PRIMARY_COMPONENT_CAP
    )


ood_negative = assignment[
    (
        assignment[
            "partition"
        ] == "ood"
    )
    & (
        assignment[
            "label"
        ] == 0
    )
]

assert int(
    ood_negative.groupby(
        "homology_component"
    ).size().max()
) <= OOD_NEG_COMPONENT_CAP


# Anchor training representative.
eval_anchor_components = (
    assignment[
        (
            assignment[
                "partition"
            ].isin(
                [
                    "validation",
                    "test",
                ]
            )
        )
        & (
            assignment[
                "label"
            ] == 1
        )
    ][
        [
            "primary_anchor_source",
            "primary_anchor_type",
            "primary_anchor",
            "homology_component",
        ]
    ]
    .drop_duplicates()
)

train_positive_components = (
    assignment[
        (
            assignment[
                "partition"
            ] == "train"
        )
        & (
            assignment[
                "label"
            ] == 1
        )
    ][
        [
            "primary_anchor_source",
            "primary_anchor_type",
            "primary_anchor",
            "homology_component",
        ]
    ]
    .drop_duplicates()
)

for anchor_tuple, g in (
    eval_anchor_components.groupby(
        [
            "primary_anchor_source",
            "primary_anchor_type",
            "primary_anchor",
        ],
        dropna=False,
    )
):
    src, typ, anchor = (
        anchor_tuple
    )

    train_match = (
        train_positive_components[
            (
                train_positive_components[
                    "primary_anchor_source"
                ] == src
            )
            & (
                train_positive_components[
                    "primary_anchor_type"
                ] == typ
            )
            & (
                train_positive_components[
                    "primary_anchor"
                ] == anchor
            )
        ]
    )

    assert len(
        train_match
    ) >= 1


# Length-policy self-checks.
length_checks = {}

eligible_positive_lengths = (
    candidate_proteins[
        "length"
    ]
    .astype(int)
    .to_numpy()
)

validation_positive_lengths = (
    assignment.loc[
        (
            assignment[
                "partition"
            ] == "validation"
        )
        & (
            assignment[
                "label"
            ] == 1
        ),
        "length",
    ]
    .astype(int)
    .to_numpy()
)

test_positive_lengths = (
    assignment.loc[
        (
            assignment[
                "partition"
            ] == "test"
        )
        & (
            assignment[
                "label"
            ] == 1
        ),
        "length",
    ]
    .astype(int)
    .to_numpy()
)

train_positive_lengths = (
    assignment.loc[
        (
            assignment[
                "partition"
            ] == "train"
        )
        & (
            assignment[
                "label"
            ] == 1
        ),
        "length",
    ]
    .astype(int)
    .to_numpy()
)


length_checks[
    "positive_validation_vs_eligible_pool"
] = ks_statistic(
    validation_positive_lengths,
    eligible_positive_lengths,
)

length_checks[
    "positive_test_vs_eligible_pool"
] = ks_statistic(
    test_positive_lengths,
    eligible_positive_lengths,
)

length_checks[
    "positive_validation_vs_test"
] = ks_statistic(
    validation_positive_lengths,
    test_positive_lengths,
)

assert (
    length_checks[
        "positive_validation_vs_eligible_pool"
    ]
    <= 0.10 + 1e-12
)

assert (
    length_checks[
        "positive_test_vs_eligible_pool"
    ]
    <= 0.10 + 1e-12
)

assert (
    length_checks[
        "positive_validation_vs_test"
    ]
    <= 0.10 + 1e-12
)


# Report-only structural comparisons.
length_checks[
    "positive_train_vs_validation_report_only"
] = ks_statistic(
    train_positive_lengths,
    validation_positive_lengths,
)

length_checks[
    "positive_train_vs_test_report_only"
] = ks_statistic(
    train_positive_lengths,
    test_positive_lengths,
)


# Hard within-partition class matching.
for partition in [
    "train",
    "validation",
    "test",
    "ood",
]:

    p_lengths = assignment.loc[
        (
            assignment[
                "partition"
            ] == partition
        )
        & (
            assignment[
                "label"
            ] == 1
        ),
        "length",
    ]

    n_lengths = assignment.loc[
        (
            assignment[
                "partition"
            ] == partition
        )
        & (
            assignment[
                "label"
            ] == 0
        ),
        "length",
    ]

    key = (
        f"{partition}_positive_vs_negative"
    )

    value = ks_statistic(
        p_lengths,
        n_lengths,
    )

    length_checks[
        key
    ] = value

    assert value <= 0.10 + 1e-12


# Source balance audit.
source_balance = {}

candidate_source_props = {
    k:
        v / candidate_pool_n
    for k, v
    in candidate_source_totals.items()
}

for partition in [
    "validation",
    "test",
]:
    pids = set(
        assignment.loc[
            (
                assignment[
                    "partition"
                ] == partition
            )
            & (
                assignment[
                    "label"
                ] == 1
            ),
            "protein_id",
        ]
    )

    counts = Counter(
        pos.loc[
            pos[
                "protein_id"
            ].isin(
                pids
            ),
            "sources",
        ].astype(str)
    )

    source_balance[
        partition
    ] = {}

    for source_name, pool_prop in (
        sorted(
            candidate_source_props.items()
        )
    ):
        observed = (
            counts[
                source_name
            ]
            / VAL_POS_TARGET
        )

        delta = (
            observed
            - pool_prop
        )

        source_balance[
            partition
        ][
            source_name
        ] = {
            "eligible_pool_fraction":
                pool_prop,
            "observed_fraction":
                observed,
            "delta":
                delta,
        }

        assert (
            abs(delta)
            <= 0.05 + 1e-12
        )


assert len(
    assignment[
        (
            assignment[
                "partition"
            ] == "ood"
        )
        & (
            assignment[
                "label"
            ] == 1
        )
    ]
) == 321

assert set(
    assignment.loc[
        (
            assignment[
                "partition"
            ] == "ood"
        )
        & (
            assignment[
                "label"
            ] == 1
        ),
        "homology_component",
    ]
) == final_ood_components

print("Builder structural self-checks: PASS")


# ============================================================
# BUILD FINAL DATASETS
# ============================================================

combined = pd.concat(
    [
        pos.copy(),
        neg.copy(),
    ],
    ignore_index=True,
    sort=False,
)

combined = combined.merge(
    components[
        [
            "protein_id",
            "old_cdhit_cluster",
            "homology_component",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
)

combined[
    "split"
] = combined[
    "protein_id"
].map(
    partition_by_id
)

ordered_columns = [
    "protein_id",
    "sequence",
    "length",
    "label",
    "homology_component",
    "old_cdhit_cluster",
    "split",
    "sources",
    "source_accessions",
    "source_record_count",
    "accession",
    "organism",
    "organism_id",
    "protein_name",
]

for col in ordered_columns:
    if col not in combined.columns:
        combined[col] = np.nan

combined = combined[
    ordered_columns
]

combined = combined.sort_values(
    "protein_id"
).reset_index(
    drop=True
)


# Component-level biological group table.
group_table = component_stats.copy()

group_table = group_table.merge(
    unified[
        [
            "homology_component",
            "amrf_mapped_proteins",
            "mapping_fraction",
            "component_lca",
            "primary_anchor_source",
            "primary_anchor_type",
            "primary_anchor",
            "primary_anchor_span_components",
        ]
    ],
    on="homology_component",
    how="left",
)

component_partition = (
    assignment.groupby(
        "homology_component"
    )["partition"]
    .first()
)

group_table[
    "partition"
] = group_table[
    "homology_component"
].map(
    component_partition
)

group_table[
    "ood_block_id"
] = group_table[
    "homology_component"
].map(
    ood_block_lookup
)


# ============================================================
# WRITE TO LOCAL STAGING ONLY
# ============================================================

with tempfile.TemporaryDirectory(
    dir=None,
    prefix="amr_v2_split_",
) as tmpdir:

    tmp = Path(tmpdir)

    staged = {}

    for partition in [
        "train",
        "validation",
        "test",
        "ood",
    ]:
        df = combined[
            combined[
                "split"
            ] == partition
        ].copy()

        path = tmp / (
            f"{partition}.tsv.gz"
        )

        write_tsv_gz_deterministic(
            df,
            path,
        )

        staged[
            f"data/processed/"
            f"{partition}.tsv.gz"
        ] = path


    assignment_out = (
        assignment.sort_values(
            "protein_id"
        )
        .reset_index(
            drop=True
        )
    )

    assignment_path = (
        tmp
        / "v2_split_assignment.tsv.gz"
    )

    write_tsv_gz_deterministic(
        assignment_out,
        assignment_path,
    )

    staged[
        "metadata/"
        "v2_split_assignment.tsv.gz"
    ] = assignment_path


    groups_path = (
        tmp
        / "v2_biological_groups.tsv.gz"
    )

    write_tsv_gz_deterministic(
        group_table.sort_values(
            "homology_component"
        ),
        groups_path,
    )

    staged[
        "metadata/"
        "v2_biological_groups.tsv.gz"
    ] = groups_path


    output_hashes = {
        name:
            sha256_file(path)
        for name, path
        in staged.items()
    }


    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    eligible_block_summary = []

    for block_i, block in enumerate(
        eligible_blocks_sorted,
        start=1,
    ):
        stats = component_stats[
            component_stats[
                "homology_component"
            ].isin(
                block
            )
        ]

        eligible_block_summary.append({
            "block_id":
                f"OOD_BLOCK_{block_i:03d}",

            "components":
                sorted(block),

            "positive_proteins":
                int(
                    stats[
                        "positives"
                    ].sum()
                ),

            "largest_component":
                int(
                    stats[
                        "proteins"
                    ].max()
                ),
        })


    manifest = {
        "schema":
            "amr_binary_study_v2_split_v1",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "generated_not_independently_audited",

        "split_seed":
            SPLIT_SEED,

        "solver":
            {
                "name":
                    "OR-Tools CP-SAT",

                "ortools_version":
                    ortools.__version__,

                "num_search_workers":
                    1,

                "random_seed":
                    SPLIT_SEED,
            },

        "design_note":
            {
                "path":
                    "MANUSCRIPT_NOTES_v2.md",

                "sha256":
                    sha256_file(
                        DESIGN_NOTE
                    ),
            },

        "input_hashes":
            {
                "positive_model_range.tsv.gz":
                    sha256_file(
                        POS_PATH
                    ),

                "negative_model_range.tsv.gz":
                    sha256_file(
                        NEG_PATH
                    ),

                "preprocessing_evidence.tar.gz":
                    sha256_file(
                        EVIDENCE_PATH
                    ),

                "ReferenceGeneCatalog.txt":
                    sha256_file(
                        AMRF_CATALOG_PATH
                    ),

                "ReferenceGeneHierarchy.txt":
                    sha256_file(
                        AMRF_HIERARCHY_PATH
                    ),

                "card.json":
                    sha256_file(
                        CARD_JSON_PATH
                    ),

                "protein_fasta_protein_homolog_model.fasta":
                    sha256_file(
                        CARD_FASTA_PATH
                    ),
            },

        "database_versions":
            {
                "AMRFinderPlus":
                    EXPECTED_AMRF_VERSION,

                "CARD":
                    EXPECTED_CARD_VERSION,
            },

        "targets":
            {
                "train":
                    {
                        "positive":
                            TRAIN_POS_EXPECTED,
                        "negative":
                            TRAIN_NEG_EXPECTED,
                    },

                "validation":
                    {
                        "positive":
                            VAL_POS_TARGET,
                        "negative":
                            VAL_NEG_TARGET,
                    },

                "test":
                    {
                        "positive":
                            TEST_POS_TARGET,
                        "negative":
                            TEST_NEG_TARGET,
                    },

                "ood":
                    {
                        "positive":
                            OOD_POS_EXPECTED,
                        "negative":
                            OOD_NEG_TARGET,
                    },
            },

        "component_caps":
            {
                "primary_evaluation":
                    PRIMARY_COMPONENT_CAP,

                "ood_negative":
                    OOD_NEG_COMPONENT_CAP,

                "ood_positive":
                    None,
            },

        "biological_grouping":
            {
                "pure_positive_components":
                    339,

                "primary_anchored_components":
                    281,

                "primary_anchored_proteins":
                    5257,

                "ood_blocks":
                    24,

                "ood_positive_components":
                    29,

                "ood_positive_proteins":
                    321,

                "common_category_min_fraction":
                    float(
                        COMMON_CATEGORY_MIN_FRACTION
                    ),

                "common_category_min_components":
                    COMMON_CATEGORY_MIN_COMPONENTS,

                "common_categories":
                    common_categories,
            },

        "positive_optimizer_objectives":
            positive_objectives,

        "negative_optimizer_objectives":
            negative_objectives,

        "length_ks":
            length_checks,

        "positive_source_balance":
            source_balance,

        "ood_blocks":
            eligible_block_summary,

        "output_hashes":
            output_hashes,

        "held_out_predictions_generated":
            {
                "test":
                    False,
                "ood":
                    False,
            },

        "independent_audit_passed":
            False,
    }


    manifest_path = (
        tmp
        / "v2_split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    staged[
        "metadata/"
        "v2_split_manifest.json"
    ] = manifest_path


    # --------------------------------------------------------
    # Commit staged artifacts to Drive only after all checks.
    # --------------------------------------------------------

    OUT_PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_METADATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    destinations = {
        "data/processed/train.tsv.gz":
            OUT_PROCESSED
            / "train.tsv.gz",

        "data/processed/validation.tsv.gz":
            OUT_PROCESSED
            / "validation.tsv.gz",

        "data/processed/test.tsv.gz":
            OUT_PROCESSED
            / "test.tsv.gz",

        "data/processed/ood.tsv.gz":
            OUT_PROCESSED
            / "ood.tsv.gz",

        "metadata/v2_split_assignment.tsv.gz":
            OUT_METADATA
            / "v2_split_assignment.tsv.gz",

        "metadata/v2_biological_groups.tsv.gz":
            OUT_METADATA
            / "v2_biological_groups.tsv.gz",

        "metadata/v2_split_manifest.json":
            OUT_METADATA
            / "v2_split_manifest.json",
    }

    for key, destination in (
        destinations.items()
    ):
        assert not destination.exists()

        shutil.copy2(
            staged[key],
            destination,
        )


# ============================================================
# FINAL PRINT SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("HOMOLOGY-AWARE SPLIT BUILDER COMPLETE")
print("=" * 72)

print("\nPartition counts:")
print(
    pd.crosstab(
        assignment["partition"],
        assignment["label"],
    )
)

print("\nLength KS statistics:")
for key, value in sorted(
    length_checks.items()
):
    print(
        f"  {key}: {value:.6f}"
    )

print("\nPositive source balance:")
for partition, values in (
    source_balance.items()
):
    print(f"  {partition}")

    for source_name, item in (
        values.items()
    ):
        print(
            "   ",
            source_name,
            "pool=",
            f"{item['eligible_pool_fraction']:.4f}",
            "observed=",
            f"{item['observed_fraction']:.4f}",
            "delta=",
            f"{item['delta']:+.4f}",
        )

print("\nOOD:")
print(
    "  biological blocks:",
    len(
        eligible_blocks_sorted
    ),
)
print(
    "  positive components:",
    len(
        final_ood_components
    ),
)
print(
    "  positive proteins:",
    OOD_POS_EXPECTED,
)

print("\nOutput SHA256:")
for name, digest in sorted(
    output_hashes.items()
):
    print(
        f"  {name}: {digest}"
    )

print(
    "\nBuilder structural checks: PASS"
)

print(
    "Independent audit: NOT YET RUN"
)

print(
    "Test predictions generated: NO"
)

print(
    "OOD predictions generated: NO"
)
