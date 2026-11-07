"""Build a reproducible, taxonomy-balanced subset of the expanded instructions."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import datasets
import typer
from datasets import Image, List
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from build_artifacts import merge_index, write_artifact, write_readme

REPO = "mdctleo/RoboJailBench"
REVISION = "582848836ba2ae19748f5307e445c2b1ac7311ce"
NAME = "RJB-Instructions-20"


def taxonomy(key: str) -> str:
    match = re.fullmatch(r"\$(.+)Example\s+\d+", Path(key).stem)
    if not match:
        raise ValueError(f"Unrecognized taxonomy key: {key}")
    return match.group(1)


def select(entries: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    seen_ids, seen_keys = set(), set()
    for entry in entries:
        if entry["dataset"] == "RJB-Instructions":
            if entry["id"] in seen_ids or entry["key"] in seen_keys:
                raise ValueError("Expanded instruction index contains duplicate IDs or keys")
            seen_ids.add(entry["id"])
            seen_keys.add(entry["key"])
            groups[taxonomy(entry["key"])].append(entry)
    if len(groups) != 18 or any(len(group) != 50 for group in groups.values()):
        raise ValueError("Expected 18 taxonomy categories of 50 instructions each")
    selected = []
    for category, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda entry: hashlib.sha256(
            ("robobench-rjb20-v1:" + entry["key"]).encode()).hexdigest())
        selected.extend(ranked[:20])
    return selected


def main(output: Path = Path("artifacts/expanded-pilot")):
    load_dotenv()
    index_file = hf_hub_download(REPO, "index.json", repo_type="dataset", revision=REVISION)
    chosen = select(json.loads(Path(index_file).read_text()))
    parquet = hf_hub_download(REPO, "data/RJB-Instructions/train.parquet",
                              repo_type="dataset", revision=REVISION)
    ds = datasets.load_dataset("parquet", data_files=parquet, split="train")
    ds = ds.cast_column("images", List(Image(decode=False)))
    positions = {id_: i for i, id_ in enumerate(ds["id"])}
    if not {entry["id"] for entry in chosen} <= positions.keys():
        raise ValueError("Pinned artifact and index disagree")
    frames = []
    for entry in chosen:
        row = ds[positions[entry["id"]]]
        frames.append({"key": entry["key"], "category": row["category"],
                       "benign": row["benign"], "malicious": row["malicious"],
                       "images": row["images"]})
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    merged = merge_index(write_artifact(NAME, frames, output), NAME, index_path)
    index_path.write_text(json.dumps(merged, indent=1))
    write_readme(sorted({entry["dataset"] for entry in merged}), output)
    manifest = {"source_repo": REPO, "source_revision": REVISION,
                "selection": "20 per taxonomy category, lowest sha256 of robobench-rjb20-v1:<key>",
                "rows": [{"source_id": e["id"], "key": e["key"],
                          "taxonomy": taxonomy(e["key"])} for e in chosen]}
    (output / "selection.json").write_text(json.dumps(manifest, indent=2))
    print(f"{NAME}: {len(frames)} rows -> {output}")


if __name__ == "__main__":
    typer.run(main)
