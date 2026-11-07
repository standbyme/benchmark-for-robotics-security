"""Evaluate generated parquets as they become available in the local collection."""

import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from main_attack_defense import cell_path, make_defense, run_cell, scene_frame
from robobench.index import load_index
from robobench.providers import effective_model
from score_experiments import main as score_experiment

TASKS = [("shawshank", name) for name in
         ("DROID", "RH20T", "ROBOVQA", "robo2vlm", "car", "RJB-Instructions-20")]
TASKS.append(("robopair", "RJB-Instructions-20"))
TASKS.extend((attack, "RJB-Instructions-20") for attack in
             ("badrobot_cd", "badrobot_cj", "badrobot_sm"))


def main(artifacts: Path = Path("artifacts/expanded-pilot"),
         model: str = "gpt-5.4-mini-2026-03-17", workers: int = 8,
         watch: bool = False, interval: int = 30, output_root: Path = Path("workdir"),
         sequence: bool = False):
    if workers < 1 or interval < 1:
        raise ValueError("workers and interval must be positive")
    load_dotenv()
    model = effective_model(model)
    defense = make_defense("no_defense", model, 5, "scene_graphs")
    index_path = artifacts / "index.json"
    pending = ({(attack, "DROID-Sequence-5") for attack in ("shawshank", "badrobot_cj", "badrobot_sm")}
               if sequence else set(TASKS))
    while pending:
        index = load_index(index_path)
        for attack, base in sorted(pending):
            artifact = f"{attack}_{base}"
            if not index.rows(base) or not index.rows(artifact):
                continue
            card = (artifacts / "README.md").read_text()
            if any(f"- config_name: {name}\n" not in card for name in (base, artifact)):
                continue
            output_dir = output_root / ("sequence-five" if sequence else
                                        "rjb20" if base == "RJB-Instructions-20" else "external-shawshank")
            for kind, name, cell_attack in (("benign", base, "clean"),
                                             ("malicious", artifact, attack)):
                frame, scenes = scene_frame(name, kind, str(artifacts))
                run_cell(defense, frame, scenes,
                         cell_path(output_dir, model, "no_defense", kind, base, cell_attack),
                         workers, f"{base}/{cell_attack}/{kind}")
            output = output_dir / ("report.json" if sequence or base == "RJB-Instructions-20"
                                   else f"{base}-report.json")
            score_experiment(base, index_path, output, output_dir, model, "no_defense")
            pending.remove((attack, base))
        if not watch or not pending:
            break
        print("waiting for " + ", ".join(f"{attack}_{base}" for attack, base in sorted(pending)), flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    typer.run(main)
