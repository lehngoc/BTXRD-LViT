from __future__ import annotations

import argparse

from src.protocol.dataset_protocol import build_ssl_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen Phase 0 D_L/D_U manifests.")
    parser.add_argument("--canonical-split", default="configs/splits/btxrd_split_seed42.csv")
    parser.add_argument("--export-manifest", default="data/exports/btxrd_preprocessed/all.csv")
    parser.add_argument("--output-dir", default="data/exports/phase0_ssl")
    parser.add_argument("--protocol-config", default="configs/protocol/phase0_ssl.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_ssl_manifests(
        canonical_split_path=args.canonical_split,
        export_manifest_path=args.export_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        protocol_config_path=args.protocol_config,
    )
    print(result)


if __name__ == "__main__":
    main()
