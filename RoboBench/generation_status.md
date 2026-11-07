# Generation status — Azure run

The original OpenAI project reached its spending limit. Its completed results
and partial trees remain under `artifacts/expanded-pilot` and
`generated_attacks/expanded`; they are not mixed into the new model-generated
attack trees. Sixteen RoboPAIR checkpoints overlapping that quota error were
reopened, with their previous entries backed up in
`workdir/generation-logs/robopair-quota-backup/`.

The new collection is `artifacts/azure-run`, and new trees are under
`generated_attacks/azure-run`. Its source is the pinned Hugging Face revision
`582848836ba2ae19748f5307e445c2b1ac7311ce`. The inherited DROID and RoboVQA
Shawshank parquets were generated before the Azure run. The inherited CJ and
SM artifacts use model-independent templates. Their provenance is recorded in
`artifacts/azure-run/inherited_artifacts.json`. All newly generated OpenAI
requests use `gpt-5.4-mini-1`; Anthropic requests use `claude-opus-5-5`.
The separate `artifacts/azure-sequence-five` collection pairs the same five
DROID sequences and template attacks with a newly generated Azure Shawshank
artifact. Its reports are written under `workdir/azure-sequence-new`.
The first 123 conceptual-deception requests were filtered before generation,
while the provider's default output limit was still in effect. That fact is
recorded in the attack configuration; subsequent requests use the limit below.
The completed Azure conceptual-deception parquet has 360 null attacks: Azure
filtered all 360 requests. The earlier direct-OpenAI parquet with 360 generated
rewrites remains in `artifacts/expanded-pilot` as a separate condition.

Azure can filter benchmark prompts. Generation records those rows as null, with
the reason in `generation_failures.json`. Evaluation records provider-filtered
requests separately and excludes them from scored response coverage. An empty
RoboPAIR result can also mean its search found no successful attack. These
outcomes must not be interpreted as model refusals or successful attacks.

The endpoint advertises 100 requests and 100,000 tokens per minute. The Azure
OpenAI client coordinates requests across local processes at 60 requests per
minute by default and observes provider reset headers. Set
`ROBOBENCH_AZURE_RPM` to a lower value if the endpoint is shared or throttles.
Shawshank and conceptual-deception calls reserve at most 1,024 output tokens;
RoboPAIR uses smaller role-specific limits. The limits are recorded in each
generated model-based attack configuration. The full searches can take many
hours at these limits.

The builders checkpoint each row. To resume a stopped Azure run, run these
commands from the RoboBench directory in separate terminals:

```sh
uv run --isolated --no-default-groups --with python-dotenv python src/build_attacks.py \
  --attack badrobot_cd --dataset RJB-Instructions-20 --artifacts artifacts/azure-run \
  --output-dir generated_attacks/azure-run --workers 4
```

```sh
ROBOBENCH_WORD2VEC=data/word2vec/GoogleNews-vectors-negative300.bin.gz \
uv run --isolated --no-default-groups --extra shawshank --with python-dotenv \
  python src/build_attacks.py --attack shawshank --dataset RJB-Instructions-20 \
  --artifacts artifacts/azure-run --output-dir generated_attacks/azure-run --workers 6
```

```sh
ROBOBENCH_WORD2VEC=data/word2vec/GoogleNews-vectors-negative300.bin.gz \
uv run --isolated --no-default-groups --extra shawshank --with python-dotenv \
  python src/build_attacks.py --attack shawshank --dataset RH20T \
  --dataset robo2vlm --dataset car --artifacts artifacts/azure-run \
  --output-dir generated_attacks/azure-run --workers 6
```

```sh
uv run --isolated --no-default-groups --with python-dotenv python src/build_attacks.py \
  --attack robopair --dataset RJB-Instructions-20 --artifacts artifacts/azure-run \
  --output-dir generated_attacks/azure-run --workers 4
```

```sh
ROBOBENCH_WORD2VEC=data/word2vec/GoogleNews-vectors-negative300.bin.gz \
uv run --isolated --no-default-groups --extra shawshank --with python-dotenv \
  python src/build_attacks.py --attack shawshank --dataset DROID-Sequence-5 \
  --artifacts artifacts/azure-sequence-five --output-dir generated_attacks/azure-sequence-five --workers 4
```

The following watchers build only complete trees and evaluate each available
parquet. Run one finalizer and one evaluator for the collection:

```sh
uv run --isolated --no-default-groups --extra sequences --with python-dotenv \
  python src/finalize_generated_attacks.py --artifacts artifacts/azure-run \
  --trees generated_attacks/azure-run --watch
```

```sh
uv run --isolated --no-default-groups --extra sequences --with python-dotenv \
  python src/evaluate_generated_attacks.py --artifacts artifacts/azure-run \
  --output-root workdir/azure-run --workers 4 --watch
```

For the sequence collection, use the same watcher commands with `--sequence`:

```sh
uv run --isolated --no-default-groups --extra sequences --with python-dotenv \
  python src/finalize_generated_attacks.py --artifacts artifacts/azure-sequence-five \
  --trees generated_attacks/azure-sequence-five --sequence --watch
```

```sh
uv run --isolated --no-default-groups --extra sequences --with python-dotenv \
  python src/evaluate_generated_attacks.py --artifacts artifacts/azure-sequence-five \
  --output-root workdir/azure-sequence-new --sequence --workers 2 --watch
```

Each tree has a generation configuration; resuming with a different deployment,
endpoint, or output-token setting is rejected. Reports are under
`workdir/azure-run`. Evaluation checkpoints every ten completed rows as well as
at the end of a cell. Per-attack reports reuse the vanilla benign observations;
the overall report counts those observations only once. Binomial intervals do
not account for correlations when attacks share a source scene. No artifact has
been pushed to Hugging Face.
