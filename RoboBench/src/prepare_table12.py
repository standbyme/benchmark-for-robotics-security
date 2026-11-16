"""Pin the published RJB-5 and external artifacts for the five-attack table."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from build_artifacts import write_readme
from robobench.ids import row_id
from robobench.index import Index

REPO = "mdctleo/RoboJailBench"
BASES = ("DROID", "ROBOVQA", "RH20T", "robo2vlm", "car", "RJB-Instructions-5")
ATTACKS = ("badrobot_cd", "badrobot_cj", "badrobot_sm", "robopair")


def main(output: Path = Path("artifacts/table12-rjb5"), revision: str | None = None):
    load_dotenv()
    revision = revision or HfApi().dataset_info(REPO).sha
    metadata = output / "source.json"
    if metadata.exists() and json.loads(metadata.read_text())["revision"] != revision:
        raise ValueError("Refusing to mix source revisions")
    remote_index = json.loads(Path(hf_hub_download(REPO, "index.json", repo_type="dataset",
                                                revision=revision)).read_text())
    by_id = {entry["id"]: entry for entry in remote_index}
    names = [name for base in BASES for name in (base, *(f"{attack}_{base}" for attack in ATTACKS))]
    names.append("shawshank_ROBOVQA")
    entries = []
    manifest = {}
    for name in names:
        source = Path(hf_hub_download(REPO, f"data/{name}/train.parquet", repo_type="dataset", revision=revision))
        attack = next((kind for kind in (*ATTACKS, "shawshank") if name.startswith(kind + "_")), None)
        base = name[len(attack) + 1:] if attack else name
        table = pq.read_table(source)
        for row in table.to_pylist():
            entry = by_id.get(row["id"])
            key = (entry["key"] if entry else
                   f"{row['category'] or base}/{Path(row['images'][0]['path']).name}")
            if row_id(name, key) != row["id"]:
                raise ValueError(f"{name}: cannot recover stable key for {row['id']}")
            entries.append({"id": row["id"], "dataset": name, "base": base,
                            "attack": attack, "target": row["target"], "key": key})
        dest = output / "data" / name / "train.parquet"
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"{dest}: source has changed")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        manifest[name] = {"rows": table.num_rows, "sha256": checksum}
        print(f"staged {name}: {table.num_rows} rows", flush=True)
    Index(entries)
    (output / "index.json").write_text(json.dumps(entries, indent=1))
    write_readme(sorted(names), output)
    metadata.write_text(json.dumps({"repo": REPO, "revision": revision, "artifacts": manifest}, indent=2))


if __name__ == "__main__":
    typer.run(main)
