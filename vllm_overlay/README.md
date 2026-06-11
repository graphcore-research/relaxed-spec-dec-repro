# vLLM Relaxed Spec-Dec Overlay

Install stock `vllm==0.20.1`, then run `python scripts/apply_vllm_overlay.py`.
The applier copies the patched Python files under `files/` into the installed
wheel before any backend imports vLLM.

This keeps CUDA extensions, vendored packages, and wheel layout identical to
the vanilla speculative-decoding job.  The overlay is intentionally small and
should be replaced by a real patched wheel if this path becomes durable.

The patches include relaxed-target rejection sampling, entropy monitoring, and
`relaxed_T_qp` bonus-token sampling for draft-model speculative decoding plus
Qwen3.5 built-in MTP. EAGLE3 and DFlash relaxed bonus-token paths remain out
of scope.
