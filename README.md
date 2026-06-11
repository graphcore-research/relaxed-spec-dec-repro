# Relaxed Spec-Dec Paper Reproduction

This repository contains the clean reproduction code for the paper experiments
on training-free relaxed speculative decoding. It is intentionally narrower
than the internal working tree: no cluster launch layer, cloud artifact sync,
plotting workflow, paper-writing workflow, or non-paper benchmarks.

## Scope

Included tasks:

- AIME24: 30 H4 AIME 2024 problems.
- GPQA Diamond: 198 questions with deterministic answer-choice shuffling.
- LiveCodeBench Code Generation Lite v6: the 175-problem non-cumulative v6
  test split.

Included model settings:

- Qwen3-32B verifier with Qwen3-0.6B drafter.
- Qwen3-32B verifier with Qwen3-8B drafter.
- Qwen3.5-27B native MTP.

Included methods:

- AR verifier baseline.
- strict speculative decoding.
- always-accept drafter baseline.
- `CACTUS`, `mentored-dec`, `spec-casc-opt`, `spec-casc-tok`, `r-fuzzy`,
  `ens`, and `spec-cont-dec`.

No generated model outputs are distributed. Reproducibility comes from the
runner, exact configs/grid, dataset preparation, scoring code, aggregation code,
and compact component profiles used for proxy speed.

## Setup

Use a fresh environment. The vLLM overlay mutates the installed vLLM Python
files, so do not apply it inside a shared environment.

```bash
uv sync
python scripts/apply_vllm_overlay.py
python scripts/prepare_data.py --tasks gpqa lcb
```

AIME24 is shipped as a checked-in 30-row fixture. GPQA may require an HF token:

```bash
export HF_TOKEN=...
python scripts/prepare_data.py --tasks gpqa
```

LCB evaluation executes generated Python. Run it in an isolated environment.

## One-Run Smoke

List a clipped debug config:

```bash
python scripts/run_paper_grid.py list \
  --preset debug \
  --tasks aime24 \
  --model-settings qwen3_0p6b_32b \
  --methods cactus \
  --draft-lengths 3 \
  --max-configs 1
```

Run it:

```bash
python scripts/run_paper_grid.py run \
  --preset debug \
  --tasks aime24 \
  --model-settings qwen3_0p6b_32b \
  --methods cactus \
  --draft-lengths 3 \
  --max-configs 1 \
  --tensor-parallel-size 1 \
  --runs-root runs/debug
```

For LCB, add local code evaluation:

```bash
python scripts/run_paper_grid.py run \
  --preset debug \
  --tasks lcb \
  --model-settings qwen3_0p6b_32b \
  --methods strict \
  --draft-lengths 3 \
  --max-configs 1 \
  --evaluate-lcb \
  --runs-root runs/debug-lcb
```

## Debug Grid

Write all clipped debug configs across paper tasks, model settings, methods,
draft lengths, and relaxation values:

```bash
python scripts/run_paper_grid.py write-configs \
  --preset debug \
  --output-dir configs/generated/debug
```

Run the clipped debug grid:

```bash
python scripts/run_paper_grid.py run \
  --preset debug \
  --output-dir configs/generated/debug \
  --runs-root runs/debug \
  --evaluate-lcb
```

Use filters such as `--tasks`, `--model-settings`, `--methods`,
`--draft-lengths`, `--first-value-only`, and `--max-configs` to run smaller
slices.

On a memory-constrained local GPU, use model overrides for mechanical smoke
tests. This does not reproduce paper numbers; it verifies the runner, overlay,
parsers, scorers, and artifact schema:

```bash
python scripts/run_paper_grid.py run \
  --preset debug \
  --tasks aime24 \
  --model-settings qwen3_0p6b_32b \
  --methods cactus \
  --draft-lengths 3 \
  --max-configs 1 \
  --override-verifier-model Qwen/Qwen3-0.6B \
  --override-draft-model Qwen/Qwen3-0.6B \
  --runs-root runs/smoke-small
```

## Full Paper Rerun

Generate full configs:

```bash
python scripts/run_paper_grid.py write-configs \
  --preset full \
  --output-dir configs/generated/full
```

Run the full grid:

```bash
python scripts/run_paper_grid.py run \
  --preset full \
  --output-dir configs/generated/full \
  --runs-root runs/full \
  --evaluate-lcb
```

Full runs are expensive. The script uses the paper sampling settings:
temperature `0.6`, top-`p=0.95`, top-`k=20`, draft lengths `{3,5,10,20}`,
standard thinking budget `32768`, and standard maximum generation length
`36864`. Qwen3.5-27B on AIME24 and LCB uses thinking budget `65536` and maximum
generation length `69632`.

## Aggregation

Write compact run summaries:

```bash
python scripts/aggregate_results.py \
  --runs-root runs/full \
  --output runs/full/summary.csv
```

Compute proxy-speed reports for completed speculative runs:

```bash
python scripts/compute_proxy_speed.py \
  --run-dirs runs/full/<spec-run-dir> \
  --ar-baseline-run-dirs runs/full/<matching-ar-run-dir> \
  --profiles profiles/paper/qwen3_32b_0p6b_b200_profile.json \
  --output runs/full/proxy_speed.json
```

The included profiles are compact B200 component profiles used by the paper's
response-length-aware proxy-speed calculation.

## Notes

- Exact bitwise reproducibility is not expected.
- Seeds are fixed within task/model/method grids.
- Model weights are downloaded from their upstream providers.
- Full benchmark data is not committed, except the small AIME24 fixture.
- The vLLM overlay targets `vllm==0.20.1`.
