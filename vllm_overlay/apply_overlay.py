"""Apply the bundled relaxed-spec-dec vLLM Python overlay.

Install the stock vLLM wheel first so native extensions and vendored backend
packages stay identical to a vanilla speculative-decoding run. This script then
copies only the patched Python files into that installed package.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
from pathlib import Path


EXPECTED_VLLM_VERSION = "0.20.1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overlay Cactus Python patches onto installed vLLM."
    )
    parser.add_argument(
        "--overlay-root",
        default=Path(__file__).resolve().parent / "files",
        type=Path,
        help="Directory containing paths relative to the installed vLLM package.",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Copy files even when installed vLLM is not 0.20.1.",
    )
    args = parser.parse_args()

    package_root, copied = apply_overlay(
        overlay_root=args.overlay_root,
        allow_version_mismatch=args.allow_version_mismatch,
    )

    print(f"Applied Cactus vLLM overlay to {package_root}")
    for relative_path in copied:
        print(f"  {relative_path}")
    return 0


def apply_overlay(
    overlay_root: Path,
    allow_version_mismatch: bool = False,
) -> tuple[Path, list[Path]]:
    installed_version = importlib.metadata.version("vllm")
    if installed_version != EXPECTED_VLLM_VERSION and not allow_version_mismatch:
        raise SystemExit(
            "Cactus overlay expects vllm=="
            f"{EXPECTED_VLLM_VERSION}, found {installed_version}. "
            "Use --allow-version-mismatch only for local diagnostics."
        )

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Could not locate installed vLLM package.")

    package_root = Path(next(iter(spec.submodule_search_locations)))
    if not overlay_root.is_dir():
        raise SystemExit(f"Overlay root does not exist: {overlay_root}")

    copied: list[Path] = []
    for source in sorted(overlay_root.rglob("*.py")):
        relative_path = source.relative_to(overlay_root)
        destination = package_root / relative_path
        if not destination.exists():
            raise SystemExit(
                "Refusing to create a new vLLM file from overlay: "
                f"{relative_path}"
            )
        shutil.copy2(source, destination)
        copied.append(relative_path)

    return package_root, copied


if __name__ == "__main__":
    raise SystemExit(main())
