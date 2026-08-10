from __future__ import annotations

import argparse

from src.protocol.locked_test import create_locked_test_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze checkpoint/config/threshold before final test.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-artifact", default=None)
    args = parser.parse_args()
    print(create_locked_test_artifact(
        checkpoint=args.checkpoint,
        config=args.config,
        selected_threshold=args.threshold,
        output_path=args.output,
        selection_artifact=args.selection_artifact,
    ))


if __name__ == "__main__":
    main()
