from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.evaluate_unet import resolve_threshold, selected_threshold_from_summary


class EvaluateThresholdTests(unittest.TestCase):
    def test_uses_selected_threshold_from_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "best.pt"
            checkpoint.touch()
            (run_dir / "best_summary.json").write_text(json.dumps({"selected_threshold": 0.65}), encoding="utf-8")

            self.assertEqual(selected_threshold_from_summary(checkpoint), 0.65)
            self.assertEqual(resolve_threshold(checkpoint, 0.5, None), (0.65, "best_summary.json"))
            self.assertEqual(resolve_threshold(checkpoint, 0.5, 0.65), (0.65, "command-line override"))


if __name__ == "__main__":
    unittest.main()
