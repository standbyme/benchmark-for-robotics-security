"""Build complete attack trees into parquets, optionally waiting for searches."""

import json
import time
from pathlib import Path

import typer

from build_artifacts import build, find_sources
from robobench.ids import frame_key
from robobench.index import load_index

TASKS = [("shawshank", base) for base in
         ("DROID", "RH20T", "robo2vlm", "car", "RJB-Instructions-20")]
TASKS.append(("robopair", "RJB-Instructions-20"))
TASKS.append(("badrobot_cd", "RJB-Instructions-20"))


def completed_keys(tree: Path, base: str) -> set[str]:
    if not tree.exists():
        return set()
    try:
        sources = find_sources(tree, base)
    except SystemExit:
        return set()
    return {frame_key(source, filename) for source, path, _ in sources
            for filename in json.loads(path.read_text())}


def main(artifacts: Path = Path("artifacts/expanded-pilot"),
         trees: Path = Path("generated_attacks/expanded"),
         watch: bool = False, interval: int = 30, sequence: bool = False):
    if interval < 1:
        raise ValueError("interval must be positive")
    index_path = artifacts / "index.json"
    pending = {("shawshank", "DROID-Sequence-5")} if sequence else set(TASKS)
    while pending:
        index = load_index(index_path)
        for attack, base in sorted(pending):
            name = f"{attack}_{base}"
            expected = {entry["key"] for entry in index.rows(base)}
            if not expected:
                continue
            tree = trees / name
            # A collection may explicitly include an already-built upstream
            # artifact without a local generated tree.
            if not tree.exists() and index.rows(name) and (artifacts / "data" / name / "train.parquet").exists():
                pending.remove((attack, base))
                continue
            got = completed_keys(tree, base)
            if got - expected:
                raise ValueError(f"{name}: generated keys not in source artifact")
            if got != expected:
                continue
            build(tree, None, artifacts, index_path)
            pending.remove((attack, base))
        if not watch or not pending:
            break
        print("waiting for " + ", ".join(f"{attack}_{base}" for attack, base in sorted(pending)), flush=True)
        time.sleep(interval)
    if pending and not watch:
        print(f"{len(pending)} attack trees are incomplete")


if __name__ == "__main__":
    typer.run(main)
