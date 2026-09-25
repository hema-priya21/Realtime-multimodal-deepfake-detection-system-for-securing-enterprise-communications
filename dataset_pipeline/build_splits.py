"""Builds subject-disjoint train/val/test splits for FF++ and Celeb-DF v2.

A video's subject_id string alone (see subject_utils.extract_subject_id) is
not enough to guarantee disjointness: a real video and a fake pair-video can
share an identity while having differently-shaped subject_id strings. This
script union-finds the individual identity tokens across every record first,
so the resulting groups are true connected components of identities -- then
splits those components, and independently re-verifies zero token overlap
before writing anything out.

Run as: python -m dataset_pipeline.build_splits
"""

import csv
import random
import sys

from dataset_pipeline.config import (
    RAW_FFPP_ROOT,
    RAW_CELEBDF_ROOT,
    SPLITS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    VIDEO_EXTENSIONS,
)
from dataset_pipeline.subject_utils import extract_subject_id, get_label


def collect_videos(root, dataset_tag):
    if not root.exists():
        print(f"WARNING: {dataset_tag} root not found, skipping: {root}")
        return []

    videos = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    print(f"Found {len(videos):,} {dataset_tag} video files under {root}")
    return videos


def build_records():
    records = []

    for path in collect_videos(RAW_FFPP_ROOT, "ffpp"):
        records.append({
            "filepath": str(path),
            "label": get_label(path, "ffpp"),
            "dataset": "ffpp",
            "subject_id": extract_subject_id(path, "ffpp"),
        })

    for path in collect_videos(RAW_CELEBDF_ROOT, "celebdf"):
        records.append({
            "filepath": str(path),
            "label": get_label(path, "celebdf"),
            "dataset": "celebdf",
            "subject_id": extract_subject_id(path, "celebdf"),
        })

    return records


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_by_identity(records):
    """Union-find over individual identity tokens so a real video and every
    fake video sharing either of its identities land in the same group."""
    uf = UnionFind()

    for record in records:
        tokens = record["subject_id"].split("+")
        for token in tokens[1:]:
            uf.union(tokens[0], token)

    groups = {}
    for record in records:
        root = uf.find(record["subject_id"].split("+")[0])
        groups.setdefault(root, []).append(record)

    return groups


def split_groups(groups):
    group_keys = list(groups.keys())
    random.Random(RANDOM_SEED).shuffle(group_keys)

    total_files = sum(len(groups[key]) for key in group_keys)
    train_target = TRAIN_RATIO * total_files
    val_target = VAL_RATIO * total_files

    split_records = {"train": [], "val": [], "test": []}
    split_group_keys = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for key in group_keys:
        group_records = groups[key]
        if counts["train"] < train_target:
            target = "train"
        elif counts["val"] < val_target:
            target = "val"
        else:
            target = "test"
        split_records[target].extend(group_records)
        split_group_keys[target].append(key)
        counts[target] += len(group_records)

    return split_records, split_group_keys


def verify_disjoint(split_records):
    """Independent re-check: derive identity tokens straight from the
    subject_id of every written record and hard-fail on any overlap."""
    split_tokens = {}
    for split_name, records in split_records.items():
        tokens = set()
        for record in records:
            tokens.update(record["subject_id"].split("+"))
        split_tokens[split_name] = tokens

    names = list(split_tokens.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_tokens[names[i]] & split_tokens[names[j]]
            if overlap:
                raise AssertionError(
                    f"SUBJECT LEAKAGE DETECTED between '{names[i]}' and "
                    f"'{names[j]}': shared identities {sorted(overlap)}"
                )


def write_split_csv(name, records):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS_DIR / f"{name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "dataset", "subject_id"])
        writer.writeheader()
        writer.writerows(records)
    return out_path


def print_summary(records, groups, split_records, split_group_keys):
    print()
    print("=" * 75)
    print("SUBJECT-DISJOINT SPLIT SUMMARY")
    print("=" * 75)
    print(f"Total video files found : {len(records):,}")
    print(f"Total unique subjects   : {len(groups):,}")
    print()

    for name in ("train", "val", "test"):
        recs = split_records[name]
        real_count = sum(1 for r in recs if r["label"] == "real")
        fake_count = sum(1 for r in recs if r["label"] == "fake")
        print(
            f"{name.upper():5s} | subjects: {len(split_group_keys[name]):5,} | "
            f"files: {len(recs):6,} (real {real_count:,} / fake {fake_count:,})"
        )
    print("=" * 75)


def main():
    records = build_records()

    if not records:
        print("\nERROR: No video files found for FF++ or Celeb-DF. Nothing to split.")
        sys.exit(1)

    groups = group_by_identity(records)
    split_records, split_group_keys = split_groups(groups)
    verify_disjoint(split_records)

    print("\nNo subject-ID overlap between train/val/test: CONFIRMED")
    print()

    for name in ("train", "val", "test"):
        out_path = write_split_csv(name, split_records[name])
        print(f"Wrote {len(split_records[name]):,} rows -> {out_path}")

    print_summary(records, groups, split_records, split_group_keys)


if __name__ == "__main__":
    main()
