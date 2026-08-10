from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_locked_test_artifact(
    *,
    checkpoint: str | Path,
    config: str | Path,
    selected_threshold: float,
    output_path: str | Path,
    selection_artifact: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    config_path = Path(config).resolve()
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Locked-test checkpoint and config must both exist.")
    artifact = {
        "mode": "final_locked",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selected_threshold": float(selected_threshold),
        "selection_artifact": str(Path(selection_artifact).resolve()) if selection_artifact else None,
    }
    Path(output_path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def require_locked_test(
    *,
    locked_final: bool,
    lock_artifact_path: str | Path | None,
    checkpoint: str | Path,
    config: str | Path,
) -> dict[str, Any]:
    if not locked_final or lock_artifact_path is None:
        raise PermissionError("Test evaluation requires --locked-final and --lock-artifact.")
    artifact_path = Path(lock_artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing locked-test artifact: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("mode") != "final_locked":
        raise PermissionError("Locked-test artifact is not in final_locked mode.")
    if Path(artifact.get("checkpoint", "")).resolve() != Path(checkpoint).resolve():
        raise PermissionError("Checkpoint differs from the frozen locked-test checkpoint.")
    if Path(artifact.get("config", "")).resolve() != Path(config).resolve():
        raise PermissionError("Config differs from the frozen locked-test config.")
    if artifact.get("config_sha256") != sha256_file(config):
        raise PermissionError("Config content changed after the test was locked.")
    if artifact.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise PermissionError("Checkpoint content changed after the test was locked.")
    if "selected_threshold" not in artifact:
        raise PermissionError("Locked-test artifact has no selected threshold.")
    return artifact
