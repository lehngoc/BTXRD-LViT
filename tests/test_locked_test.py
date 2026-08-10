from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.protocol.locked_test import create_locked_test_artifact, require_locked_test


class LockedTestTests(unittest.TestCase):
    def test_test_requires_explicit_matching_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, lock = root / "best.pt", root / "config.yaml", root / "final_lock.json"
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("seed: 42\n", encoding="utf-8")
            create_locked_test_artifact(
                checkpoint=checkpoint, config=config, selected_threshold=0.6, output_path=lock
            )
            artifact = require_locked_test(
                locked_final=True, lock_artifact_path=lock, checkpoint=checkpoint, config=config
            )
            self.assertEqual(artifact["selected_threshold"], 0.6)
            with self.assertRaises(PermissionError):
                require_locked_test(
                    locked_final=False, lock_artifact_path=lock, checkpoint=checkpoint, config=config
                )
            config.write_text("seed: 43\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                require_locked_test(
                    locked_final=True, lock_artifact_path=lock, checkpoint=checkpoint, config=config
                )


if __name__ == "__main__":
    unittest.main()
