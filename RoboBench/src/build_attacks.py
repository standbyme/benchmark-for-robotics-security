"""Generate an attack over a vanilla artifact, as a reviewable raw-layout tree.

For every frame of the base artifact the attack
(``robobench.attacks.ATTACKS[<attack>]``) rewrites the malicious instruction
and, for attacks that touch pixels, the frame. The result is a copy of the
base's raw tree with the malicious text replaced, written as
``<output-dir>/<attack>_<base>/`` in the layout ``build_artifacts.py`` reads:

    <attack>_<base>/<base>_instr.json + <attack>_<base>/dataset/<frame>          one source
    <attack>_<base>/<source>/<source>_instr.json + .../<source>/dataset/<frame>  categorized base

The instruction file maps each frame path to ``[malicious, benign]``, benign
copied from the artifact unchanged; an attack that found nothing leaves the
malicious text empty. Frames keep the names recorded in the package index.
Review the file and the frames, then build the artifact:

    uv run python src/build_artifacts.py generated_attacks/<attack>_<base>

Frames already in the instruction file are skipped, so a run can be stopped and
restarted. The base artifact is read from ``--artifacts``, the Hugging Face
dataset by default or a local artifact directory.

Usage:
    uv run python src/build_attacks.py --attack robopair --dataset DROID
    uv run python src/build_attacks.py --attack badrobot_cj
    uv run python src/build_attacks.py --attack badrobot_cd --dataset DROID --limit 3
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from main_attack_defense import ARTIFACTS, BASES
from robobench.attacks import ATTACKS, AttackResult
from robobench.index import load_index
from robobench.providers import is_content_filtered
from robobench.scenes import Scenes, load_scenes
from robobench.attacks.robopair import DEFAULT_ATTACK_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_TARGET_MODEL


def record_generation_config(tree: Path, config: dict) -> None:
    """Refuse to resume a model-generated tree under different settings."""
    tree.mkdir(parents=True, exist_ok=True)
    config_path = tree / "generation_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if any(existing.get(key) != value for key, value in config.items()):
            raise ValueError(f"{config_path}: refusing to mix generation model settings")
        config = {**existing, **config}
    elif any(tree.rglob("*_instr.json")):
        raise ValueError(f"{tree}: existing checkpoints lack generation settings")
    config_path.write_text(json.dumps(config, indent=2))


def build_tree(attack, base: str, scenes: Scenes, index, output_dir: Path,
               limit=None, workers: int = 1) -> Path:
    """Run ``attack`` over ``scenes`` into ``<output_dir>/<attack.name>_<base>``."""
    tree = output_dir / f"{attack.name}_{base}"
    table = scenes.dataset.remove_columns("images").to_pandas()

    # Shawshank draws its few-shot examples from the base's own malicious
    # instructions, the nearest faithful reconstruction of the pool upstream
    # withheld. Other attacks absorb this through ``**extra`` and ignore it.
    case_pool = table["malicious"].dropna().unique().tolist()
    if limit is not None:
        table = table.head(limit)

    instructions: dict[Path, dict] = {}  # instruction file -> its entries
    failure_path = tree / "generation_failures.json"
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}

    def instruction_file(source: str) -> Path:
        source_dir = tree if source == base else tree / source
        path = source_dir / f"{source}_instr.json"
        if path not in instructions:
            instructions[path] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        return path

    if workers < 1:
        raise ValueError("workers must be at least 1")
    jobs = []
    for row in table.itertuples(index=False):
        source, filename = index[row.id]["key"].split("/", 1)
        instr_path = instruction_file(source)
        frame_path = instr_path.parent / "dataset" / filename
        if frame_path.as_posix() in instructions[instr_path]:
            continue
        jobs.append((row, instr_path, frame_path))

    def run(job):
        row, _, _ = job
        source_images = scenes.image_bytes(row.id)
        raw_images = [image["bytes"] for image in source_images]
        if not raw_images or any(raw is None for raw in raw_images):
            raise ValueError(f"{row.id}: scene must contain embedded image bytes")
        failure = None
        try:
            result = attack.attack(text=row.malicious, images=raw_images, case_pool=case_pool)
        except Exception as error:
            if not is_content_filtered(error):
                raise
            result = AttackResult(malicious=None)
            failure = "provider_content_filter"
        if result.malicious is None and failure is None:
            failure = "no_attack_generated"
        result_images = result.images if result.images is not None else raw_images
        if len(result_images) != len(raw_images):
            raise ValueError(f"{row.id}: attack changed the number of frames")
        return job, source_images, result, result_images, failure

    def save(completed):
        (row, instr_path, frame_path), source_images, result, result_images, failure = completed
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(result_images[0])
        extra_paths = []
        for position, raw in enumerate(result_images[1:], start=1):
            suffix = Path(source_images[position]["path"] or "").suffix or ".png"
            extra_path = frame_path.parent / "sequences" / frame_path.name / f"{position:06d}{suffix}"
            extra_path.parent.mkdir(parents=True, exist_ok=True)
            extra_path.write_bytes(raw)
            extra_paths.append(extra_path.as_posix())
        pair = [result.malicious or "", row.benign]
        if extra_paths:
            pair.append(extra_paths)
        if failure:
            failures[row.id] = {"key": index[row.id]["key"], "reason": failure}
        else:
            failures.pop(row.id, None)
        if failures or failure_path.exists():
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_tmp = failure_path.with_suffix(".json.tmp")
            failure_tmp.write_text(json.dumps(failures, indent=2), encoding="utf-8")
            failure_tmp.replace(failure_path)
        instructions[instr_path][frame_path.as_posix()] = pair
        tmp_path = instr_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(instructions[instr_path], indent=4), encoding="utf-8")
        tmp_path.replace(instr_path)

    if workers == 1:
        for job in tqdm(jobs, desc=tree.name):
            save(run(job))
    elif jobs:
        # Initialize the lazy image column before multiple threads access it.
        scenes.image_bytes(jobs[0][0].id)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            saved = set()
            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc=tree.name):
                    save(future.result())
                    saved.add(future)
            except BaseException:
                for future in futures:
                    future.cancel()
                # Let in-flight calls finish and checkpoint successful results,
                # even if another call failed or the operator interrupted us.
                pool.shutdown(wait=True, cancel_futures=True)
                for future in futures:
                    if future not in saved and not future.cancelled() and future.exception() is None:
                        save(future.result())
                raise

    return tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=sorted(ATTACKS))
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--artifacts", default=ARTIFACTS,
                        help="Hugging Face dataset or local artifact directory")
    parser.add_argument("--output-dir", default="generated_attacks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel attack searches; each result is checkpointed")
    parser.add_argument("--robopair-attack-model", default=DEFAULT_ATTACK_MODEL)
    parser.add_argument("--robopair-target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--robopair-judge-model", default=DEFAULT_JUDGE_MODEL)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    if args.attack == "robopair":
        attack = ATTACKS[args.attack](attack_model=args.robopair_attack_model,
                                      target_model=args.robopair_target_model,
                                      judge_model=args.robopair_judge_model)
    else:
        attack = ATTACKS[args.attack]()
    local_index = Path(args.artifacts) / "index.json"
    index = load_index(local_index if local_index.is_file() else None)
    output_dir = Path(args.output_dir)

    for base in args.datasets or BASES:
        if args.attack in {"robopair", "shawshank", "badrobot_cd"}:
            tree = output_dir / f"{args.attack}_{base}"
            if args.attack == "robopair":
                settings = attack.search_args("")
                config = {"attack_model": attack.attack_model, "target_model": attack.target_model,
                          "judge_model": attack.judge_model, "iterations": attack.n_iterations,
                          "robot": attack.robot,
                          "attack_max_n_tokens": settings.attack_max_n_tokens,
                          "target_max_n_tokens": settings.target_max_n_tokens,
                          "judge_max_n_tokens": settings.judge_max_n_tokens}
            elif args.attack == "shawshank":
                config = {"attack_model": attack._attacker.name, "vlm_model": attack._vlm.name,
                          "judge_model": attack._judge.name, "iterations": attack.n_iterations,
                          "is_vlm": attack.is_vlm}
            else:
                config = {"rewrite_model": attack.model}
            config["provider"] = "azure" if os.getenv("AZURE_OPENAI_API_BASE") else "openai"
            if config["provider"] == "azure":
                endpoint = os.environ["AZURE_OPENAI_API_BASE"].rstrip("/")
                config["endpoint_sha256"] = hashlib.sha256(endpoint.encode()).hexdigest()
                if args.attack in {"shawshank", "badrobot_cd"}:
                    config["max_completion_tokens"] = 1024
            record_generation_config(tree, config)
        scenes = load_scenes(base, args.artifacts)
        tree = build_tree(attack, base, scenes, index, output_dir, args.limit, args.workers)
        print(f"done: {tree}")

    print(f"review the instruction files, then build with src/build_artifacts.py {output_dir}/<attack>_<base>")


if __name__ == "__main__":
    main()
