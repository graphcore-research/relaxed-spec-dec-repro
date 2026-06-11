# Reproducing The Paper Experiments

The intended reproduction path is:

1. Install dependencies in a fresh environment.
2. Apply the bundled vLLM overlay.
3. Prepare GPQA and LCB data locally.
4. Run clipped debug configs to validate the environment.
5. Run the full grid if compute budget permits.
6. Aggregate quality and acceptance summaries.
7. Join speculative runs with the included component profiles for proxy speed.

The debug preset is not a substitute for the paper numbers. It exists to verify
that every task, method family, parameter value, parser, scorer, and output
schema can run end-to-end from a fresh checkout.

The full preset is the paper reproduction path. It uses the final task/model
scope and parameter grids described in `configs/paper/README.md`.
