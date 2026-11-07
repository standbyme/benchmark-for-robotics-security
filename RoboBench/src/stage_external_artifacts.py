"""Copy the pinned external vanilla parquets into a local artifact collection."""

import hashlib
import json
import shutil
from pathlib import Path

import typer
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from build_artifacts import merge_index, write_readme
from prepare_expanded_instructions import REPO, REVISION
from robobench.index import load_index

BASES = ("DROID", "RH20T", "ROBOVQA", "robo2vlm", "car")
EXISTING_ATTACKS = ("shawshank_ROBOVQA",)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main(output: Path = Path("artifacts/expanded-pilot")):
    load_dotenv()
    source_index = Path(hf_hub_download(REPO, "index.json", repo_type="dataset", revision=REVISION))
    remote_entries = json.loads(source_index.read_text())
    index_path = output / "index.json"
    for name in (*BASES, *EXISTING_ATTACKS):
        relative = f"data/{name}/train.parquet"
        source = Path(hf_hub_download(REPO, relative, repo_type="dataset", revision=REVISION))
        selected = [entry for entry in remote_entries if entry["dataset"] == name]
        if not selected and name in EXISTING_ATTACKS:
            # The pinned HF index omits this parquet; the shipped index has it.
            # Accept it only when every id matches the actual parquet.
            selected = load_index().rows(name)
        actual_ids = pq.read_table(source, columns=["id"])["id"].to_pylist()
        if len(actual_ids) != len(selected) or set(actual_ids) != {entry["id"] for entry in selected}:
            raise ValueError(f"{name}: index ids do not match the pinned parquet")
        if not selected:
            raise ValueError(f"Pinned source index lacks {name}")
        destination = output / relative
        if destination.exists() and digest(source) != digest(destination):
            raise ValueError(f"{destination} exists with different content")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        merged = merge_index(selected, name, index_path)
        index_path.write_text(json.dumps(merged, indent=1))
        write_readme(sorted({entry["dataset"] for entry in merged}), output)
        print(f"staged {name}: {len(selected)} rows")


if __name__ == "__main__":
    typer.run(main)
