#!/usr/bin/env python3
"""Utility for constructing the T2 retune v1.1c 50-run experiment matrix.

The script combines the primary ACR-W retune configuration with the
exploratory SGLD study and expands them into a list of run descriptors. Each
run is annotated with a deterministic seed, warmup window, cosine decay phase
and stability targets mirroring the validated synthesis from v1.1.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "experiments"
DEFAULT_T2_CONFIG = EXPERIMENT_DIR / "exp_t2_acrw_v1_1c.yaml"
DEFAULT_SGLD_CONFIG = EXPERIMENT_DIR / "exp_exploration_sgld_v1_1c.yaml"


@dataclass(frozen=True)
class RunDescriptor:
    """Flattened description of a single run in the 50-run matrix."""

    run_id: int
    seed: int
    profile: str
    warmup_steps: int
    cosine_cycles: int
    lr_base: float
    rho: float
    gradient_norm_limit: float
    stability_target: float
    stability_requirement: float

    def to_dict(self) -> Dict[str, float]:
        """Return the descriptor as a serialisable dictionary."""
        data = asdict(self)
        data["profile"] = self.profile
        return data


def _parse_scalar(token: str):
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if token.startswith("\"") and token.endswith("\""):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    try:
        if token.startswith("0") and token != "0" and not token.startswith("0."):
            raise ValueError
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def _simple_yaml_load(text: str) -> Dict:
    """Parse a restricted YAML subset covering the project configuration."""
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, stripped))

    root: Dict = {}
    stack: List[tuple[int, object]] = [(-1, root)]

    for index, (indent, content) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("List item encountered without a list parent")
            item = content[2:].strip()
            if not item:
                new_obj: Dict = {}
                parent.append(new_obj)
                stack.append((indent, new_obj))
            else:
                parent.append(_parse_scalar(item))
            continue

        if ":" not in content:
            raise ValueError(f"Unable to parse line: {content}")

        key, remainder = content.split(":", 1)
        key = key.strip()
        value_part = remainder.strip()

        if value_part:
            value = _parse_scalar(value_part)
            if isinstance(parent, dict):
                parent[key] = value
            else:
                raise ValueError("Cannot assign key-value pair inside a list item without mapping")
        else:
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if next_line and next_line[0] > indent and next_line[1].startswith("- "):
                new_container: object = []
            else:
                new_container = {}
            if isinstance(parent, dict):
                parent[key] = new_container
            else:
                raise ValueError("List items must contain mappings before nested keys")
            stack.append((indent, new_container))

    return root


def load_yaml(path: Path) -> Dict:
    """Load YAML from *path* using PyYAML when available, otherwise fallback."""
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # pragma: no cover - configuration file
            raise SystemExit(f"Failed to parse YAML file {path}: {exc}") from exc
        return data or {}
    return _simple_yaml_load(text)


def _format_scalar(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def dump_yaml(data) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False)

    lines: List[str] = []

    def _emit(obj, indent: int) -> None:
        prefix = " " * indent
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    _emit(value, indent + 2)
                else:
                    lines.append(f"{prefix}{key}: {_format_scalar(value)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    _emit(item, indent + 2)
                else:
                    lines.append(f"{prefix}- {_format_scalar(item)}")
        else:
            lines.append(f"{prefix}{_format_scalar(obj)}")

    _emit(data, 0)
    return "\n".join(lines)


def build_run_matrix(
    seeds: Sequence[int],
    t2_config: Dict,
    sgld_config: Dict,
) -> List[RunDescriptor]:
    """Create a list of run descriptors using the provided configs."""
    warmup_steps = int(
        t2_config.get("optimizer", {})
        .get("learning_rate", {})
        .get("warmup", {})
        .get("steps", 600)
    )
    lr_base = float(
        t2_config.get("optimizer", {}).get("learning_rate", {}).get("base", 0.5)
    )
    rho = float(t2_config.get("optimizer", {}).get("rho", 0.8))
    gradient_norm_limit = float(
        t2_config.get("training", {}).get("gradient_norm_target", 0.5)
    )
    stability_target = float(
        t2_config.get("training", {}).get("stability_target", 0.90)
    )
    stability_requirement = float(
        t2_config.get("training", {}).get("stability_run_threshold", 0.70)
    )

    cosine_period = int(sgld_config.get("sampler", {}).get("cosine_schedule", {}).get("period", 15000))

    descriptors: List[RunDescriptor] = []
    for index, seed in enumerate(seeds, start=1):
        cycles = max(1, round(cosine_period / warmup_steps))
        profile = f"t2_acrw_v1_1c_seed{seed}"
        descriptor = RunDescriptor(
            run_id=index,
            seed=int(seed),
            profile=profile,
            warmup_steps=warmup_steps,
            cosine_cycles=cycles,
            lr_base=lr_base,
            rho=rho,
            gradient_norm_limit=gradient_norm_limit,
            stability_target=stability_target,
            stability_requirement=stability_requirement,
        )
        descriptors.append(descriptor)
    return descriptors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the T2 v1.1c run matrix")
    parser.add_argument(
        "--t2-config",
        type=Path,
        default=DEFAULT_T2_CONFIG,
        help="Path to the primary ACR-W configuration file.",
    )
    parser.add_argument(
        "--sgld-config",
        type=Path,
        default=DEFAULT_SGLD_CONFIG,
        help="Path to the exploratory SGLD configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(range(1001, 1051)),
        help="Explicit seed list. Defaults to 50 deterministic seeds.",
    )
    parser.add_argument(
        "--format",
        choices={"json", "yaml"},
        default="json",
        help="Output format for the generated matrix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    t2_config = load_yaml(args.t2_config)
    sgld_config = load_yaml(args.sgld_config)

    descriptors = build_run_matrix(args.seeds, t2_config, sgld_config)

    if args.format == "json":
        payload = json.dumps([desc.to_dict() for desc in descriptors], indent=2)
        print(payload)
    else:
        payload = dump_yaml([desc.to_dict() for desc in descriptors])
        print(payload)


if __name__ == "__main__":
    main()
