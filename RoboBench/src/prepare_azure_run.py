"""Create an isolated Azure experiment collection from reviewed local inputs."""

import json
import shutil
from pathlib import Path

import typer

from build_artifacts import write_readme
from robobench.index import Index

INHERITED = (
    "DROID", "RH20T", "ROBOVQA", "robo2vlm", "car", "RJB-Instructions-20",
    "shawshank_DROID", "shawshank_ROBOVQA",
    "badrobot_cj_RJB-Instructions-20", "badrobot_sm_RJB-Instructions-20",
)
SEQUENCE_INHERITED = ("DROID-Sequence-5", "badrobot_cj_DROID-Sequence-5",
                      "badrobot_sm_DROID-Sequence-5")


def main(source: Path = Path("artifacts/expanded-pilot"),
         output: Path = Path("artifacts/azure-run"), sequence: bool = False):
    if sequence:
        if source == Path("artifacts/expanded-pilot"):
            source = Path("artifacts/sequence-five")
        if output == Path("artifacts/azure-run"):
            output = Path("artifacts/azure-sequence-five")
    inherited = SEQUENCE_INHERITED if sequence else INHERITED
    if source.resolve() == output.resolve():
        raise ValueError("Azure collection must be separate from the source collection")
    existing_index = output / "index.json"
    if existing_index.exists():
        raise FileExistsError(f"{existing_index} exists; refusing to overwrite an experiment")
    entries = json.loads((source / "index.json").read_text())
    selected = [entry for entry in entries if entry["dataset"] in inherited]
    Index(selected)
    if set(inherited) != {entry["dataset"] for entry in selected}:
        raise ValueError("Source collection is missing an inherited artifact")
    for name in inherited:
        src = source / "data" / name
        dest = output / "data" / name
        if dest.exists():
            raise FileExistsError(f"{dest} already exists")
        dest.mkdir(parents=True)
        shutil.copy2(src / "train.parquet", dest / "train.parquet")
        if (src / "generation_config.json").exists():
            shutil.copy2(src / "generation_config.json", dest / "generation_config.json")
    metadata_name = "provenance.json" if sequence else "selection.json"
    shutil.copy2(source / metadata_name, output / metadata_name)
    existing_index.write_text(json.dumps(selected, indent=1))
    write_readme(sorted(inherited), output)
    provenance = {
        "inherited_source": str(source),
        "source_revision": (json.loads((source / metadata_name).read_text())[0]["source_revision"]
                            if sequence else json.loads((source / metadata_name).read_text())["source_revision"]),
        "inherited": list(inherited),
        "note": ("CJ and SM are model-independent templates. The sequence frames come from the "
                 "pinned DROID release." if sequence else
                 "DROID and RoboVQA Shawshank parquets were generated before this Azure run; "
                 "CJ and SM are model-independent templates. New model-generated attacks "
                 "are built separately with per-tree generation_config.json."),
    }
    (output / "inherited_artifacts.json").write_text(json.dumps(provenance, indent=2))
    print(f"staged {len(inherited)} inherited artifacts in {output}")


if __name__ == "__main__":
    typer.run(main)
