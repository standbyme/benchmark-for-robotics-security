# RoboBench

Benchmark for jailbreak attacks and defenses on vision-language robot planners.
A scene is one or more camera frames with a malicious and a benign instruction;
attacks rewrite the malicious instruction and, for visual attacks, the frame;
defenses screen the request; scorers turn accept/reject labels into ASR, SR, UR
and SU-HM.

Scenes are released as Hugging Face artifacts (dataset `mdctleo/RoboJailBench`),
one config per artifact. A vanilla artifact (`DROID`, `ROBOVQA`, `RH20T`,
`robo2vlm`, `car`, `RJB-Instructions`) holds the original instructions; an
attack artifact (`<attack>_<base>`, e.g. `robopair_DROID`) holds the same frames
with the attack's rewritten instruction. Columns are `id`, `dataset`,
`category`, `images`, `benign`, `malicious`, `target`.

## Install

```
uv sync
```

API keys are read from the environment / `.env`. For Azure, set
`AZURE_OPENAI_API_BASE`, `AZURE_ANTHROPIC_API_BASE`, and `AZURE_API_KEY`.
When the corresponding base is set, OpenAI calls use the deployment
`gpt-5.4-mini-1` and Anthropic calls use `claude-opus-5-5`; the Azure settings
take precedence over direct-provider keys. The Anthropic base may be either
the `/anthropic` endpoint or the full `/anthropic/v1/messages` URL.

Without Azure bases, direct calls use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
or `GOOGLE_API_KEY` (also accepted: `GOOGLE_AI_STUDIO_KEY`). `HF_TOKEN` is
required to read a private artifact repo and to push artifacts. Optional extras:
`anthropic`, `roboguard`, `scenegraph`, `shawshank` (see `pyproject.toml`).

Azure OpenAI requests share a local rate limiter, defaulting to 60 requests per
minute and honoring provider reset headers. Set `ROBOBENCH_AZURE_RPM` lower if
the endpoint is shared. Provider content-filter blocks are recorded as missing
requests, with reasons and coverage, rather than counted as model refusals.
Azure-generated artifacts are kept separately from artifacts generated with the
historical OpenAI models; see `generation_status.md`.

## Layout

```
src/
  dataset/generate_prompt_from_img.py   write a dataset's <base>_instr.json from its frames
  dataset/import_droid_sequences.py      import annotated windows from DROID episodes
  prepare_expanded_instructions.py     build the pinned 20-per-category subset
  stage_external_artifacts.py          stage pinned external parquet artifacts
  build_attacks.py                      run an attack over a vanilla artifact -> reviewable raw tree
  build_artifacts.py                    build one HF parquet artifact from a raw tree
  finalize_generated_attacks.py        build complete generated attack trees
  evaluate_generated_attacks.py        evaluate completed local attack artifacts
  score_experiments.py                 score custom and visual attack results with a local index
  build_scene_graphs.py                 build the scene graphs RoboGuard needs
  main_attack_defense.py                run defenses over attack/benign prompts (ASR, UR)
  resolve_unknowns.py                   judge-label responses the keyword parser cannot
  analysis_attack_defense.py            score: ASR, SR, UR, SU-HM and the LaTeX table
  main.py, analysis.py                  plain (no attack, no defense) malicious-vs-benign baseline
  backfill_ids.py                       one-off: add artifact ids to pre-artifact pickles
  robobench/                            the package: load_scenes, label, score; attacks, defenses, models

data/<name>/                raw source trees: <name>/dataset/<frame>, <name>_instr.json
generated_attacks/<attack>_<base>/   raw attack trees from build_attacks.py (reviewable before build)
artifacts/                  built artifacts: data/<name>/train.parquet, index.json, README.md (HF card)
scene_graphs/<source>/<frame stem>.json   per-scene graphs for RoboGuard
workdir/                    experiment output: model responses and scores
```

All commands run from this directory with `uv run python`.

---

# Pipeline A — Produce artifacts

Maintainer side: turn an attack into a released HF artifact. The vanilla base
artifacts already exist on Hugging Face, so this starts from an attack.

### 1. Generate an attack (once per attack per base)

```
uv run python src/build_attacks.py --attack badrobot_cj --dataset DROID
```

`build_attacks.py` reads the base artifact (from HF, or `--artifacts <dir>` for a
local one), rewrites the malicious instruction per frame — and the frame itself
for visual attacks — and writes a reviewable raw tree
`generated_attacks/<attack>_<base>/`. Attacks: `badrobot_cd`, `badrobot_cj`,
`badrobot_sm`, `robopair`, `shawshank`. Omit `--dataset` to run every base;
`--limit N` is a smoke test. A run resumes from the frames already written.
`--workers N` runs independent searches concurrently and checkpoints each
finished row. Do not run two builders for the same output tree at once.

The expanded RJB-Instructions release has 50 examples in each of 18 detailed
taxonomy categories. To generate a reproducible, balanced 20-per-category
subset, pinned to the source revision:

```
uv run --no-default-groups --extra sequences --with python-dotenv python \
  src/prepare_expanded_instructions.py
```

This writes `artifacts/expanded-pilot/data/RJB-Instructions-20/train.parquet`,
its local index, and `selection.json`, which records all source IDs and the
selection rule. Pass `--dataset RJB-Instructions-20 --artifacts
artifacts/expanded-pilot` to `build_attacks.py` and build each output tree into
the same directory with `--output-dir artifacts/expanded-pilot --index-path
artifacts/expanded-pilot/index.json`. The separate subset name gives it distinct
IDs and avoids replacing the 900-row source artifact.

RoboPAIR's historical `gpt-4` and `gpt-3.5-turbo` defaults have been retired
([OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)).
For a new Azure run, the configured deployment is used for all three roles:

```
uv run --no-default-groups --with python-dotenv python src/build_attacks.py \
  --attack robopair --dataset RJB-Instructions-20 --artifacts artifacts/azure-run \
  --output-dir generated_attacks/azure-run --workers 8 \
  --robopair-attack-model gpt-5.4-mini-1 \
  --robopair-target-model gpt-5.4-mini-1 \
  --robopair-judge-model gpt-5.4-mini-1
```

The builder records the model settings in the raw tree and refuses to resume
that tree with different settings. Results from these replacement models are a
new experiment, not directly equivalent to historical RoboPAIR artifacts. A
search with no successful jailbreak is recorded with a null malicious text;
review failures and coverage before evaluation.

To stage the pinned external vanilla parquets in the same collection and build
the requested expanded attack trees once their keys match their bases:

```
uv run --no-default-groups --extra sequences --with python-dotenv python \
  src/stage_external_artifacts.py
uv run --no-default-groups --extra sequences --with python-dotenv python \
  src/finalize_generated_attacks.py --watch
```

The finalizer will not build a partial tree. `--watch` waits and checks again;
without it, the command builds complete trees and reports how many remain.
Use only one finalizer for a collection because it updates the shared index.
The staging command also copies the existing RoboVQA Shawshank artifact. At
the pinned revision, its rows are absent from the remote index, so staging
uses the shipped index only after checking its IDs against the parquet.
`src/evaluate_generated_attacks.py --watch` can be run separately to evaluate
each completed Shawshank artifact and the expanded-set RoboPAIR artifact with
the no-defense model, resuming existing response files and writing per-dataset
JSON reports. It skips missing attack instructions and reports coverage.

### 2. Build the artifact

```
uv run python src/build_artifacts.py generated_attacks/badrobot_cj_DROID
```

Input: the reviewable raw tree from step 1 (review it first). Output:
`artifacts/data/<attack>_<base>/train.parquet`, with `artifacts/index.json`, the
shipped `src/robobench/index.json`, and the HF card `artifacts/README.md`
regenerated. The directory name is the artifact name. Push to Hugging Face to
release (`hf upload mdctleo/RoboJailBench ...`).

### 3. Scene graphs for RoboGuard (once per base)

```
uv run python src/build_scene_graphs.py
```

RoboGuard plans over a scene graph, not pixels. This reads the vanilla artifacts
from HF and writes `scene_graphs/<source>/<frame stem>.json`, where the runner
looks. Scenes that already have a graph are skipped. `--dataset D` limits it.

> For a multi-frame scene, graph generation sees all frames and produces one
> aggregate graph. Its temporal fidelity has not been validated.

### Ordered-frame pilot

The checked-in archives contain stills, not recoverable trajectories. For a
separate, small DROID pilot, download the following files from the pinned
revision of the upstream LeRobot release:

```
uv run --no-default-groups --extra sequences --with python-dotenv hf download \
  lerobot/droid_100 --repo-type dataset --revision 87301a2d2e99340e2010c9ef0f1d8e780b08aaf9 \
  --local-dir data/droid_100 meta/info.json meta/episodes/chunk-000/file-000.parquet \
  videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4
uv run --no-default-groups --extra sequences --with python-dotenv python \
  src/dataset/import_droid_sequences.py data/droid_100 \
  examples/droid_sequence_annotations.json data/DROID-Sequence
uv run --no-default-groups --with python-dotenv python src/build_artifacts.py \
  data/DROID-Sequence --output-dir artifacts/sequence-pilot \
  --index-path artifacts/sequence-pilot/index.json
cp data/DROID-Sequence/provenance.json artifacts/sequence-pilot/provenance.json
uv run --no-default-groups --with python-dotenv python src/build_attacks.py \
  --attack badrobot_cj --dataset DROID-Sequence --artifacts artifacts/sequence-pilot \
  --output-dir generated_attacks/sequence-pilot
uv run --no-default-groups --with python-dotenv python src/build_artifacts.py \
  generated_attacks/sequence-pilot/badrobot_cj_DROID-Sequence \
  --output-dir artifacts/sequence-pilot --index-path artifacts/sequence-pilot/index.json
uv run --no-default-groups --with python-dotenv python src/main_attack_defense.py \
  --artifacts artifacts/sequence-pilot --dataset DROID-Sequence \
  --defense no_defense --workers 1 --limit 1 --output-dir workdir/sequence-pilot
```

The annotation file contains one manually written pair, not a representative
benchmark. Extend it with reviewed instruction pairs and strictly increasing,
episode-relative frame offsets. The importer records the episode, camera,
offsets, frame rate, source video and pinned revision in `provenance.json`.
Keep that sidecar with any release: the current parquet schema does not embed
it. The raw instruction entry stores `[malicious, benign, [additional frame
paths...]]`, with the key naming the first frame. The parquet `images` list
preserves this order. Shawshank optimizes and overlays the first frame, retaining
the remaining frames unchanged; text attacks retain all frames. The target
receives all frames in order, but no explicit timestamps. See
`sequential_frames_and_design_review.md` for the evidence and limitations.
The five-episode exploratory run uses `examples/droid_sequence_5_annotations.json`
with the same importer and the output directory `data/DROID-Sequence-5`.
`src/score_experiments.py` scores a local runner output with its local index,
including visual attacks and custom dataset names. For the five-episode run:

```
uv run --no-default-groups --extra sequences --with python-dotenv python \
  src/score_experiments.py DROID-Sequence-5 artifacts/sequence-five/index.json \
  workdir/sequence-five/report.json --input-dir workdir/sequence-five
```

The report reuses each clean benign response for every attack on that scene;
those repeated rows are correlated, so do not treat its pooled overall metric
as independent observations.

For Shawshank, use `--extra shawshank` and `--with python-dotenv` in the attack
command and set `ROBOBENCH_WORD2VEC` to the GoogleNews vectors for normal-sized
instruction pools. A one-row pilot uses the built-in fallback examples. Build
the generated tree into the same local artifact directory before evaluation.
Use a separate `--index-path` for pilots to avoid changing the shipped index.

---

# Pipeline B — Use artifacts for experiments

## B1. Downstream collaborators

You have a model and want a RoboBench score. Load an artifact, run your model,
hand the responses back; the package labels and scores them. See
`downstream_bencharmk_design.md` for the full API.

```python
import robobench

scenes = robobench.load_scenes("robopair_DROID")   # one artifact; images are lazy PIL
responses = [
    {"id": s["id"], "type": t, "response": my_model(s["images"], s[t])}
    for s in scenes.rows for t in ("benign", "malicious")
]
report = robobench.score(responses)   # labels (keyword + judge) then scores; needs OPENAI_API_KEY
```

Input: your `[{id, type, response}]` rows (list, DataFrame, or CSV/JSONL path),
from one artifact or many. Output: a JSON-serializable report — per-artifact and
overall ASR, SR, UR, SU-HM with standard errors and 95% CIs, plus coverage. The
judge is pinned to the paper's version for direct OpenAI; Azure uses the
configured `gpt-5.4-mini-1` deployment. The effective model is recorded in the
report, and Azure-judged results are a separate condition.

## B2. Reproduce the paper

### Run attacks against defenses

```
uv run python src/main_attack_defense.py --model gpt-5.4-mini-2026-03-17
```

Input: the attack and vanilla artifacts from HF. Runs each defense
(`no_defense`, `google_prompt`, `roboguard`) over the malicious prompts (ASR) and
the clean benign goals (UR). Output:
`workdir/attack_defense/<model>/{malicious,benign}/<dataset>/<attack>/<defense>.pkl`,
each row carrying the response and a parsed `is_accepted` label. Options:
`--dataset` / `--defense` (repeatable, default all), `--workers`, `--limit`.
Re-running resumes by `id`.

RoboGuard does not depend on the target model, so it runs once under its own path
(`workdir/attack_defense/roboguard/...`). It needs the `roboguard` extra, the
`spot` system install (`src/robobench/_vendor/README.md`), and the scene graphs
from step A4 (`--scene-graphs`, default `scene_graphs`).

### Resolve unlabeled responses

```
uv run python src/resolve_unknowns.py --model gpt-5.4-mini-2026-03-17
```

Some answers are neither a clear accept nor reject token. This re-applies the
keyword parser and sends the remainder to the judge, writing `is_accepted` back
into the response files and marking which rows the judge resolved.
`--input-dir workdir/plain` runs it over the baseline instead.

### Score

```
uv run python src/analysis_attack_defense.py --model gpt-5.4-mini-2026-03-17 \
    --model-label "GPT 5.4 Mini" --output workdir/attack_defense/gpt54mini.tex
```

Prints per-artifact rates and writes the LaTeX table. `--verify` cross-checks the
RoboGuard combination.

## Plain baseline (no attack, no defense)

```
uv run python src/main.py --model gpt-5.4-mini-2026-03-17
uv run python src/resolve_unknowns.py --input-dir workdir/plain --model gpt-5.4-mini-2026-03-17
uv run python src/analysis.py
```

`main.py` sends each frame's raw malicious and benign instruction to one target
model and writes `workdir/plain/<model>/<dataset>/{benign,malicious}.pkl` (the
same columns as the attack-defense cells, so `resolve_unknowns.py` and
`analysis.py` read them directly). `analysis.py` writes per-dataset metrics and
LaTeX tables to `workdir/plain/analysis/`.
