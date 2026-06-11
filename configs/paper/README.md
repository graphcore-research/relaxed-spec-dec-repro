# Paper Grid

Tasks:

- `aime24`: AIME 2024, 30 examples, 48 repeats in full mode.
- `gpqa`: GPQA Diamond, 198 examples, 6 repeats in full mode.
- `lcb`: LiveCodeBench Code Generation Lite v6, 175 examples, 6 repeats in full
  mode.

Model settings:

- `qwen3_0p6b_32b`: Qwen3-0.6B drafter, Qwen3-32B verifier.
- `qwen3_8b_32b`: Qwen3-8B drafter, Qwen3-32B verifier.
- `qwen35_mtp_27b`: Qwen3.5-27B native MTP.

Draft lengths:

- `N_draft in {3, 5, 10, 20}`.

Relaxation grids:

- `cactus`: `delta in {0.10, 0.25, 1.0, 10.0}`.
- `ens`: verifier weight in `{0.95, 0.90, 0.80}`.
- `spec_casc_opt`: alpha in `{-0.10, 0.0, 0.05}`.
- `spec_casc_tok`: alpha in `{0.0, 0.35, 0.80}`.
- `mentored_dec`: alpha in `{0.95, 0.70, 0.30}`.
- `spec_cont_dec`: alpha in `{0.10, 0.50, 1.0}`.
- `r_fuzzy`: JS threshold in `{0.05, 0.20, 0.30}`.

Baselines:

- `ar`: autoregressive verifier.
- `strict`: strict speculative decoding.
- `always_accept`: drafter-only always-accept baseline, implemented as the
  ensemble target with verifier weight `0.0`.

Generate configs with:

```bash
python scripts/run_paper_grid.py write-configs --preset debug
python scripts/run_paper_grid.py write-configs --preset full
```
