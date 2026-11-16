# GPT-5.4-Mini five-attack evaluation

Evaluation and the paper update are complete: all 108 response cells passed
the completeness checks, Table 12 now contains all 18 dataset/defense rows and
five ASR columns, and the paper compiles with the Shawshank citation resolved.
All 138 tests pass. Upload remains blocked by the token permissions described
below. Results are in `workdir/table12/results/table12.json` and the updated
manuscript is `../6aef50c3cbcd361b1f701639/main.pdf`.

Shawshank has full coverage in every defense condition: 100 rows per external
dataset and 90 RJB-5 rows. Its no-defense ASR is 0% on DROID, 2% on RoboVQA,
6% on RH20T, 1% on Robo2VLM, 0% on PhysicalAI AV, and 0% on RJB-5. The RJB-5
Google Prompt result is 1/90; its RoboGuard result is 0/90. No provider-filtered
target or guard responses remain in this run. Ten non-binary target outputs
were resolved with the existing GPT-5.4-Mini judge procedure; none remain
unknown. RoboGuard produced 1,082 no-verdict outcomes across 3,488 inputs;
those are retained as fail-open and disclosed in the appendix.

The published RoboPAIR inputs have 52 null malicious instructions across the
six datasets (9 RoboVQA, 3 RH20T, 13 PhysicalAI AV, 27 RJB-5). Their conditional
denominators are printed in the table; they are not counted as model refusals.

This run uses the published `RJB-Instructions-5` configuration (90 rows), the
five external datasets (100 rows each), and all five attacks under No Defense,
Google Prompt, and RoboGuard. The source revision is
`a645680dcc06521a467d76b7661e9a427f5fe464` of `mdctleo/RoboJailBench`.
RJB-20 is excluded from this run and its paper table.

Inputs and source checksums are in `artifacts/table12-rjb5/source.json`.
`prepare_table12.py` pins and verifies the source parquets. The external
Shawshank artifacts are reused after comparing vanilla images and instructions;
the one missing car attack was regenerated. The RJB-5 Shawshank artifact was
generated afresh using `gpt-5.4-mini-1`. All 590 visual attack rows contain a
changed image and preserve the original instruction pair; the validation is
recorded in `artifacts/table12-rjb5/validation.json`.

Responses are checkpointed separately under `workdir/table12/results` for each
defense. Existing external scene graphs are retained. Ninety missing RJB-5 maps
were generated with GPT-5.4-Mini using JSON output and a 4,096-token budget;
the unavailable ER-1.6 endpoint is not used. RoboGuard runs with the installed
Spot 2.15.1 Python 3.14 bindings, while target evaluation uses the project's
locked Python 3.13 environment. Both use the configured Azure deployment.

Requests share a 600 RPM limiter; the endpoint advertises 5,000 RPM and
5 million TPM. A health check succeeded after several calls stalled; those
workers were resumed from saved checkpoints with a 60-second request timeout.
Transport errors stop/resume a run, provider filters are excluded and counted,
and genuine RoboGuard no-verdict outcomes remain fail-open as specified in the
paper. RoboGuard malicious success additionally requires the same source row's
no-defense GPT response to accept. Joins use artifact IDs, not row positions.

The reporting command requires all 108 response cells to be complete before
updating the paper. It recomputes ASR, pooled five-attack SR, UR, SU-HM, and
standard errors, and prints coverage when an evaluated denominator is smaller
than the source population. Published text attacks can themselves have missing
malicious instructions; these are not treated as target refusals.

Run commands from `RoboBench` with `UV_CACHE_DIR=/tmp/robobench-uv-cache`,
`HF_HOME` pointing to `workdir/table12/hf`, `ROBOBENCH_AZURE_RPM=600`, and
`ROBOBENCH_AZURE_TIMEOUT=60`. Use `uv run --no-project --python
workdir/table12/.venv/bin/python python src/complete_table12.py` followed by:

- `evaluate no_defense --workers 20`
- `evaluate google_prompt --workers 20`
- `report --write-paper` after all response cells finish
- `upload` to publish the validated Shawshank artifacts and merged index/card

For `evaluate roboguard --workers 12`, use
`workdir/table12/guard-venv/bin/python` and set
`PYTHONPATH=src:/home/htsai/usr/lib/python3.14/site-packages` plus
`LD_LIBRARY_PATH=/home/htsai/usr/lib`.

The artifact upload was attempted but rejected with HTTP 403: the current
fine-grained `HF_TOKEN` grants only `repo.content.read` for the `hongyucai`
account. A token backed by write access to `mdctleo/RoboJailBench` is required.
The prepared release is in `workdir/table12/release`; only attack parquets,
their generation metadata, and the merged dataset card/index are uploaded.
