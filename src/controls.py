"""Prototype-compatible controls wrapper.

This file exists to mirror the owner-requested prototype layout while still
allowing imports such as ``src.controls.length_control`` to resolve to the
active package implementation in ``src/controls/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__path__ = [str(Path(__file__).with_name("controls"))]

from src.controls.length_control import load_and_control_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype-compatible length-control wrapper")
    parser.add_argument("--input", required=True, help="Input processed spans CSV")
    parser.add_argument("--output", required=True, help="Output controlled CSV")
    parser.add_argument("--target-tokens", type=int, default=96)
    args = parser.parse_args()

    _, stats = load_and_control_dataset(
        input_path=args.input,
        output_path=args.output,
        target_tokens=args.target_tokens,
    )
    print(f"Controlled dataset saved to {args.output}")
    print(f"Controlled mean tokens: {stats['controlled_token_stats']['mean']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
