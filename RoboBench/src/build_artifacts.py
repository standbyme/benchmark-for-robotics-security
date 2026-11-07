"""Build one RoboBench artifact from a directory in the raw layout.

The directory's name is the artifact name: a base (``DROID``,
``RJB-Instructions``) or ``<attack>_<base>`` (``robopair_DROID``). Two layouts
are read, chosen by what is in the directory:

    <dir>/<base>_instr.json + <dir>/dataset/<frame>              one source, the base
    <dir>/<source>/<source>_instr.json + <dir>/<source>/dataset/  one source per subfolder;
                                                                   ``source`` fills ``category``

An instruction file maps a frame path (relative to the project directory) to
``[malicious, benign]`` or ``[malicious, benign, [additional frame paths]]``;
``generate_prompt_from_img.py`` writes it for raw data
and ``build_attacks.py`` for attacks. Every frame listed becomes one row. An
empty malicious text is kept as null and reported, so an attack artifact has
one row per frame of its base. The ``target`` column of an attack artifact
comes from the attack class in ``robobench.attacks``.

Output:
    artifacts/data/<name>/train.parquet   rewritten from scratch
    artifacts/index.json                  [{id, dataset, base, attack, target, key}] of every artifact
    src/robobench/index.json              copy of the index, shipped in the package
    artifacts/README.md                   dataset card, one config per artifact in the index

The index keeps the rows of every other artifact and replaces this one's.

Usage:
    uv run python src/build_artifacts.py data/external_datasets/DROID
    uv run python src/build_artifacts.py data/RJB-Instructions
    uv run python src/build_artifacts.py generated_attacks/robopair_DROID
    uv run python src/build_artifacts.py data/external_datasets/DROID --limit 3   # smoke test
"""

import argparse
import json
import shutil
from pathlib import Path

import datasets
from datasets import Dataset, Features, Image, Value

datasets.disable_progress_bars()

try:  # datasets >= 4 names the list feature ``List``; older versions ``Sequence``
    from datasets import List as ListFeature
except ImportError:  # pragma: no cover
    from datasets import Sequence as ListFeature

from main_attack_defense import split_artifact
from robobench.attacks import ATTACKS
from robobench.ids import frame_key, row_id
from robobench.index import Index

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts"
PACKAGE_INDEX = ROOT / "src/robobench/index.json"

FEATURES = Features({
    "id": Value("string"),
    "dataset": Value("string"),
    "category": Value("string"),
    "images": ListFeature(Image()),
    "benign": Value("string"),
    "malicious": Value("string"),
    "target": Value("string"),
})


def find_sources(directory: Path, base: str) -> list[tuple[str, Path, bool]]:
    """``(source, instruction file, categorized)`` for each source of ``directory``."""
    single = directory / f"{base}_instr.json"
    if single.exists():
        return [(base, single, False)]
    sources = [
        (sub.name, sub / f"{sub.name}_instr.json", True)
        for sub in sorted(directory.iterdir())
        if sub.is_dir() and (sub / f"{sub.name}_instr.json").exists()
    ]
    if not sources:
        raise SystemExit(f"{directory}: no instruction file for {base} and no source subfolders")
    return sources


def load_frames(directory: Path, name: str, limit) -> list[dict]:
    """One record per frame of the artifact: key, source, texts, bytes."""
    _, base = split_artifact(name)
    frames = []
    empty = []
    for source, instr_path, categorized in find_sources(directory, base):
        data = json.loads(instr_path.read_text(encoding="utf-8"))
        items = list(data.items())
        if limit is not None:
            items = items[:limit]
        for filename, pair in items:
            extra_paths = pair[2] if len(pair) > 2 else []
            if not isinstance(extra_paths, list) or not all(isinstance(p, str) for p in extra_paths):
                raise ValueError(f"{instr_path}: additional frames must be a list of paths")
            images = [{"bytes": (ROOT / path).read_bytes(), "path": Path(path).name}
                      for path in [filename, *extra_paths]]
            frame_name = Path(filename).name
            malicious = pair[0]
            if malicious is None or not str(malicious).strip():
                malicious = None
                empty.append(frame_name)
            frames.append({
                "key": frame_key(source, filename),
                "category": source if categorized else None,
                "malicious": malicious,
                "benign": pair[1],
                "images": images,
            })

    keys = [frame["key"] for frame in frames]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{name}: duplicate frame keys")
    if empty:
        # Frames with no malicious text keep their row, with a null instruction,
        # so an attack artifact has one row per frame of its base.
        print(f"  WARNING {name}: {len(empty)} empty malicious texts kept as null: "
              + ", ".join(empty))
    return frames


def write_artifact(name: str, frames: list[dict], out_dir: Path, target_override=None) -> list[dict]:
    """Write ``data/<name>/train.parquet`` from scratch; return the index entries."""
    attack, base = split_artifact(name)
    target = target_override or (ATTACKS[attack].target if attack is not None else None)
    rows = []
    entries = []
    for frame in frames:
        id_ = row_id(name, frame["key"])
        rows.append({
            "id": id_,
            "dataset": name,
            "category": frame["category"],
            "images": frame["images"],
            "benign": frame["benign"],
            "malicious": frame["malicious"],
            "target": target,
        })
        entries.append({
            "id": id_,
            "dataset": name,
            "base": base,
            "attack": attack,
            "target": target,
            "key": frame["key"],
        })

    artifact_dir = out_dir / "data" / name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    path = artifact_dir / "train.parquet"
    Dataset.from_list(rows, features=FEATURES).to_parquet(str(path))
    print(f"  {name}: {len(rows)} rows -> {path}")
    return entries


def merge_index(entries: list[dict], name: str, index_path: Path) -> list[dict]:
    """The index with ``name``'s rows replaced by ``entries``; duplicate ids abort."""
    existing = []
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
    merged = [entry for entry in existing if entry["dataset"] != name] + entries
    Index(merged)  # raises on duplicate ids
    return merged


def write_readme(names: list[str], out_dir: Path) -> None:
    lines = ["---", "configs:"]
    for name in names:
        lines += [
            f"- config_name: {name}",
            "  data_files:",
            "  - split: train",
            f"    path: data/{name}/train.parquet",
        ]
    lines += [
        "---",
        "",
        "# RoboBench",
        "",
        "Benchmark artifacts for jailbreak attacks on vision-language robot planners.",
        "One config per artifact; every artifact is self-contained (images included).",
        "Vanilla artifacts carry the original malicious and benign instruction per",
        "frame; `<attack>_<base>` artifacts carry the attack's rewritten malicious",
        "instruction with the same frames and benign instruction.",
        "",
        "Columns: `id` (uuid, unique across the repo), `dataset`, `category`,",
        "`images`, `benign`, `malicious`, `target`.",
        "",
    ]
    card = out_dir / "README.md"
    temporary = card.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(card)


def build(directory: Path, limit, out_dir: Path, index_path: Path) -> list[dict]:
    """Build the artifact named after ``directory``; return the merged index."""
    name = directory.resolve().name
    print(name)
    frames = load_frames(directory, name, limit)
    config_path = directory / "generation_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    attack, _ = split_artifact(name)
    target = config.get("target_model") if attack == "robopair" else None
    entries = write_artifact(name, frames, out_dir, target_override=target)
    if config:
        shutil.copyfile(config_path, out_dir / "data" / name / "generation_config.json")
    failures = directory / "generation_failures.json"
    if failures.exists():
        shutil.copyfile(failures, out_dir / "data" / name / "generation_failures.json")
    index = merge_index(entries, name, index_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_index = out_dir / "index.json"
    temporary = artifact_index.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=1), encoding="utf-8")
    temporary.replace(artifact_index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_index.resolve() != index_path.resolve():
        temporary = index_path.with_suffix(index_path.suffix + ".tmp")
        shutil.copyfile(artifact_index, temporary)
        temporary.replace(index_path)
    write_readme(sorted({entry["dataset"] for entry in index}), out_dir)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path,
                        help="dataset directory in the raw layout; its name is the artifact name")
    parser.add_argument("--limit", type=int, default=None,
                        help="frames per source folder (smoke test)")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--index-path", type=Path, default=PACKAGE_INDEX,
                        help="index to update; use a separate path for experiments")
    args = parser.parse_args()

    index = build(args.directory, args.limit, args.output_dir, args.index_path)
    names = {entry["dataset"] for entry in index}
    print(f"{len(names)} artifacts, {len(index)} index rows -> {args.output_dir}; "
          f"index copied to {args.index_path}")


if __name__ == "__main__":
    main()
