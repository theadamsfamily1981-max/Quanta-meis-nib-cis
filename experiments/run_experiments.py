#!/usr/bin/env python3
"""Lightweight experiment runner (skeleton).

Discovers YAML configs under experiments/configs, executes registered
"experiments" (dummy placeholders), and writes artifacts/summary.json.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Callable

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover - fallback to tiny loader
    yaml = None


def _fallback_yaml_load(text: str) -> Dict[str, Any]:
    """Extremely small YAML subset parser.

    Supports `key: value` pairs with scalar values (str/int/float/bool).
    Sufficient for the provided example configs when PyYAML is unavailable.
    """

    def convert(value: str) -> Any:
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            return value[1:-1]
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = convert(value)
    return data

# --------------------------- Registry -------------------------------------
ExperimentFn = Callable[[Dict[str, Any]], Dict[str, Any]]
_REGISTRY: Dict[str, ExperimentFn] = {}


def register(name: str) -> Callable[[ExperimentFn], ExperimentFn]:
    def deco(fn: ExperimentFn) -> ExperimentFn:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate experiment name: {name}")
        _REGISTRY[name] = fn
        return fn
    return deco

# ----------------------- Dummy Experiments --------------------------------
@register("mam_smoke")
def exp_mam_smoke(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder: validate MAM routing API surface.
    Returns toy metrics so the runner has something to aggregate.
    """
    # TODO: import and call real MAM once implemented
    batch_size = int(cfg.get("batch_size", 2))
    routed = True  # pretend the route() call succeeded
    return {
        "name": "mam_smoke",
        "ok": routed,
        "samples": batch_size,
        "ece": 0.12,  # placeholder
        "brier": 0.21,
    }


@register("nib_gate_smoke")
def exp_nib_gate_smoke(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder: validate EVNI gate call shape/returns."""
    threshold = float(cfg.get("threshold", 0.3))
    # TODO: evni_gate(x, threshold)
    gate_called = threshold > 0
    return {
        "name": "nib_gate_smoke",
        "ok": gate_called,
        "ece": 0.10,
        "brier": 0.18,
    }


# -------------------------- Runner ----------------------------------------
@dataclass
class RunResult:
    name: str
    ok: bool
    metrics: Dict[str, float]


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _fallback_yaml_load(text)


def find_configs(root: Path) -> List[Path]:
    return sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))


def run_one(experiment: str, cfg: Dict[str, Any]) -> RunResult:
    if experiment not in _REGISTRY:
        raise KeyError(f"Experiment '{experiment}' not registered. Known: {list(_REGISTRY)}")
    out = _REGISTRY[experiment](cfg)
    name = out.get("name", experiment)
    ok = bool(out.get("ok", True))
    metrics = {k: float(v) for k, v in out.items() if isinstance(v, (int, float))}
    return RunResult(name=name, ok=ok, metrics=metrics)


def aggregate(results: List[RunResult]) -> Dict[str, Any]:
    if not results:
        return {"count": 0, "ok": False}
    count = len(results)
    ok = all(r.ok for r in results)
    # simple averages over numeric metrics present
    sums: Dict[str, float] = {}
    n: Dict[str, int] = {}
    for r in results:
        for k, v in r.metrics.items():
            sums[k] = sums.get(k, 0.0) + v
            n[k] = n.get(k, 0) + 1
    avgs = {k: (sums[k] / max(1, n[k])) for k in sums}
    return {"count": count, "ok": ok, "avg": avgs}


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quanta experiment runner")
    p.add_argument("--config", type=Path, help="Path to a single YAML config", nargs="?")
    p.add_argument("--all", action="store_true", help="Run all YAML under experiments/configs")
    p.add_argument("--out", type=Path, default=Path("artifacts/summary.json"), help="Summary output path")
    p.add_argument("--configs-dir", type=Path, default=Path(__file__).parent / "configs", help="Configs directory")
    args = p.parse_args(argv)

    cfg_paths: List[Path] = []
    if args.config and args.all:
        p.error("Use either --config or --all, not both")
    if args.config:
        cfg_paths = [args.config]
    elif args.all:
        cfg_paths = find_configs(args.configs_dir)
    else:
        p.error("Provide --config <file> or --all")

    results: List[RunResult] = []
    for path in cfg_paths:
        cfg = load_yaml(path)
        name = cfg.get("name") or path.stem
        experiment = cfg.get("experiment")
        if not experiment:
            print(f"[WARN] Missing 'experiment' key in {path}")
            continue
        try:
            res = run_one(experiment, cfg)
            print(f"[OK] {name}: {res.ok} | metrics={res.metrics}")
            results.append(res)
        except Exception as e:
            print(f"[ERR] {name}: {e}")

    summary = aggregate(results)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary -> {out_path}\n{json.dumps(summary, indent=2)}")
    return 0 if summary.get("ok", False) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
