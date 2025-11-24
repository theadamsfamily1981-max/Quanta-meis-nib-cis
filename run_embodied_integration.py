"""Entry point for the GRTES Phase III embodied integration launch kit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from benchmark_suite import (
    CIFARAdapter,
    GLUEBaseline,
    default_hardware_profile,
)
from benchmark_suite.metrics import MetricRegistry, registry_to_payload
from benchmark_suite.metrics.serialization import merge_payload_with_metadata


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase3_results.json"),
        help="Destination path for the aggregated benchmark results.",
    )
    parser.add_argument(
        "--skip-glue",
        action="store_true",
        help="Disable the GLUE baseline evaluation.",
    )
    parser.add_argument(
        "--skip-cifar",
        action="store_true",
        help="Disable the CIFAR adapter evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the payload without writing it to disk.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _ensure_output_directory(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def run(argv: Iterable[str] | None = None) -> Dict[str, object]:
    """Execute the benchmark pipeline and return the payload."""

    args = _parse_args(argv)
    registry = MetricRegistry()
    reports: List[Dict[str, object]] = []

    if not args.skip_glue:
        glue_report = GLUEBaseline().run()
        registry.register(glue_report)
        reports.append(glue_report.to_dict())

    if not args.skip_cifar:
        cifar_report = CIFARAdapter().run()
        registry.register(cifar_report)
        reports.append(cifar_report.to_dict())

    hardware_profile = default_hardware_profile()
    hardware_summary = hardware_profile.summary()

    payload = registry_to_payload(registry)
    payload["hardware"] = hardware_summary

    payload = merge_payload_with_metadata(
        payload,
        metadata={
            "reports": reports,
            "skipped": {
                "glue": bool(args.skip_glue),
                "cifar": bool(args.skip_cifar),
            },
        },
    )

    if not args.dry_run:
        _ensure_output_directory(args.output)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    return payload


def main() -> None:
    payload = run()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    main()
