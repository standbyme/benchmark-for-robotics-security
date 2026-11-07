"""Score local attack-defense cells, including visual and custom datasets."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import typer

from robobench import score
from robobench.ids import row_id
from robobench.index import load_index
from robobench.providers import effective_model
from robobench.scoring import blocks_from_counts, type_counts


def response(row, id_: str, kind: str) -> dict:
    accepted = row.is_accepted
    label = None if accepted is None or pd.isna(accepted) else bool(accepted)
    source = None if label is None else ("judge" if getattr(row, "judge_resolved", False) else "keyword")
    raw = row.raw_response
    raw = None if raw is None or pd.isna(raw) else str(raw)
    return {"id": id_, "type": kind, "response": raw,
            "label": label, "label_source": source}


def provider_filtered(row) -> bool:
    value = getattr(row, "provider_filtered", False)
    return False if value is None or pd.isna(value) else bool(value)


def main(dataset: str, index_path: Path, output: Path,
         input_dir: Path = Path("workdir/attack_defense"),
         model: str | None = None, defense: str = "no_defense"):
    model = model or effective_model("gpt-5.4-mini-2026-03-17")
    index = load_index(index_path)
    root = input_dir / model.replace("/", "_")
    benign_path = root / "benign" / dataset / "clean" / f"{defense}.pkl"
    benign = pd.read_pickle(benign_path)
    attack_names = sorted({entry["dataset"] for entry in index.entries
                           if entry.get("base") == dataset and entry.get("attack")})
    rows = []
    filtered = {name: {"benign": 0, "malicious": 0} for name in [dataset, *attack_names]}
    for row in benign.itertuples(index=False):
        if provider_filtered(row):
            filtered[dataset]["benign"] += 1
            for name in attack_names:
                attacked_id = row_id(name, index[row.id]["key"])
                if attacked_id in index:
                    filtered[name]["benign"] += 1
            continue
        rows.append(response(row, row.id, "benign"))
        for name in attack_names:
            attacked_id = row_id(name, index[row.id]["key"])
            if attacked_id in index:
                rows.append(response(row, attacked_id, "benign"))
    for name in attack_names:
        attack = index.rows(name)[0]["attack"]
        path = root / "malicious" / dataset / attack / f"{defense}.pkl"
        if path.exists():
            for row in pd.read_pickle(path).itertuples(index=False):
                if provider_filtered(row):
                    filtered[name]["malicious"] += 1
                else:
                    rows.append(response(row, row.id, "malicious"))
    report = score(rows, index=index)
    # Benign answers are copied to attack datasets for per-attack comparisons.
    # Count the original observations only once in the combined summary.
    overall_counts = {
        "benign": type_counts(row["label"] for row in rows
                              if row["type"] == "benign" and index[row["id"]]["dataset"] == dataset),
        "malicious": type_counts(row["label"] for row in rows if row["type"] == "malicious"),
    }
    report["overall"] = {"counts": overall_counts, **blocks_from_counts(overall_counts)}
    generation_failures = {}
    for name in attack_names:
        path = index_path.parent / "data" / name / "generation_failures.json"
        if path.exists():
            reasons = (item["reason"] for item in json.loads(path.read_text()).values())
            generation_failures[name] = dict(Counter(reasons))
    report["meta"].update({"target_model": model, "defense": defense,
                           "base_dataset": dataset,
                           "provider_filtered": filtered,
                           "generation_failures": generation_failures,
                           "benign_observations": "vanilla frames, reused for each attack and counted once overall",
                           "interval_caveat": "Malicious observations can share source scenes across attacks; binomial intervals do not account for this dependence."})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"wrote {output}; artifacts: {len(report['datasets'])}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    typer.run(main)
