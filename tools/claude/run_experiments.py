#!/usr/bin/env python3

import argparse, json, os, time
from pathlib import Path
from typing import List, Dict

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = object  # soft dependency

# Local imports (relative paths)
try:
    from tools.claude.schemas import ExperimentSpec, ExperimentResult, result_schema_markdown
    from tools.claude.prompts import SYSTEM_PROMPT, USER_TEMPLATE
except Exception:
    # Fallback when running as a script without package install
    import sys
    sys.path.append(str(Path(__file__).parents[2]))
    from tools.claude.schemas import ExperimentSpec, ExperimentResult, result_schema_markdown
    from tools.claude.prompts import SYSTEM_PROMPT, USER_TEMPLATE


CATEGORIES: Dict[str, List[Dict]] = {
    "data": [
        {"id": "DATA-01", "title": "Label integrity checks", "goal": "Detect mislabels via small-loss + confusion auditing"},
        {"id": "DATA-02", "title": "Leakage probes", "goal": "Detect target leakage via permutation tests"},
        {"id": "DATA-03", "title": "Imbalance remedies", "goal": "SMOTE vs focal loss comparison"},
        {"id": "DATA-04", "title": "Missingness stress", "goal": "MNAR/MAR imputers vs baseline"},
        {"id": "DATA-05", "title": "Train/val/test hygiene", "goal": "Near-duplicate & overlap detection"}
    ],
    "ood": [
        {"id": "OOD-01", "title": "Energy vs MSP", "goal": "Compare OOD scores Δ with T-scaling"},
        {"id": "OOD-02", "title": "Mahalanobis features", "goal": "Feature-space OOD decision"},
        {"id": "OOD-03", "title": "Ensemble OOD", "goal": "Combine scores for robustness"},
        {"id": "OOD-04", "title": "Corruptions", "goal": "CIFAR-C style Δ under noise"},
        {"id": "OOD-05", "title": "Threshold tuning", "goal": "γ selection via ROC/PR"}
    ],
    "selective": [
        {"id": "SEL-01", "title": "Acc@80% implementation", "goal": "Compute coverage threshold + accuracy"},
        {"id": "SEL-02", "title": "Conformal prediction", "goal": "Set prediction sets with coverage guarantees"},
        {"id": "SEL-03", "title": "Corr(conf,correct)", "goal": "Maximize confidence-accuracy correlation"},
        {"id": "SEL-04", "title": "Abstention policy", "goal": "Evaluate business cost curve vs coverage"},
        {"id": "SEL-05", "title": "Calibration-aware selection", "goal": "Platt vs Temp vs Isotonic for selection"}
    ],
    "calibration": [
        {"id": "CAL-01", "title": "T-scaling fit", "goal": "Grid/opt search for T on val NLL"},
        {"id": "CAL-02", "title": "Isotonic", "goal": "Compare ECE/Brier improvements"},
        {"id": "CAL-03", "title": "Dirichlet calibration", "goal": "Multiclass reliability"},
        {"id": "CAL-04", "title": "Reliability diagrams", "goal": "Plot + compute ECE/MCE"}
    ],
    "optimization": [
        {"id": "OPT-01", "title": "LR sweep", "goal": "Find sane LR band quickly"},
        {"id": "OPT-02", "title": "Batch size sweep", "goal": "Stability/time trade-off"},
        {"id": "OPT-03", "title": "Regularization tuning", "goal": "Dropout/weight decay grid"},
        {"id": "OPT-04", "title": "Optimizer compare", "goal": "AdamW vs Lion vs LAMB"}
    ],
    "cis": [
        {"id": "CIS-01", "title": "PAD→τ monotonicity", "goal": "Verify τ increases with arousal and decreases with negative pleasure"},
        {"id": "CIS-02", "title": "PAD noise robustness", "goal": "Stability of τ under noise σ∈{0,0.1,0.2}"},
        {"id": "CIS-03", "title": "Routing entropy vs τ", "goal": "Measure MoE/retrieval entropy across τ"},
        {"id": "CIS-04", "title": "Top-k retrieval gating", "goal": "k∈{5,10,20} under PAD gating"},
        {"id": "CIS-05", "title": "Latency impact", "goal": "p50 latency change with gating on/off"}
    ],
    "meta": [
        {"id": "META-01", "title": "Introspection consistency", "goal": "Stability of self-reports across prompts"},
        {"id": "META-02", "title": "Identity drift", "goal": "Track embedding drift over sessions"},
        {"id": "META-03", "title": "Uncertainty gates", "goal": "Tail-risk abstention policy"}
    ],
}


def build_specs(max_total: int | None = None) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    for cat, items in CATEGORIES.items():
        for item in items:
            spec = ExperimentSpec(
                id=item["id"],
                category=cat,
                title=item["title"],
                goal=item["goal"],
                inputs={},
                outputs=["metrics.json"],
                constraints=["≤60s runtime", "≤2GB RAM"],
                acceptance_criteria=["Return JSON schema with fenced python code"]
            )
            specs.append(spec)
    if max_total:
        specs = specs[:max_total]
    return specs


def call_claude(spec: ExperimentSpec, model: str = "claude-3-5-sonnet") -> dict:
    """Call Anthropic Claude with system+user prompts.
    Returns the parsed JSON dict.
    """
    import json as _json
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if not client.api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var")

    user = USER_TEMPLATE.format(
        id=spec.id, category=spec.category, title=spec.title, goal=spec.goal,
        inputs=_json.dumps(spec.inputs), outputs=_json.dumps(spec.outputs),
        constraints=_json.dumps(spec.constraints), acceptance=_json.dumps(spec.acceptance_criteria)
    )

    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.2,
        system=SYSTEM_PROMPT + "

JSON Schema Example:

" + result_schema_markdown(),
        messages=[{"role": "user", "content": user}]
    )

    content = msg.content[0].text if getattr(msg, 'content', None) else str(msg)
    return json.loads(content)


def main():
    ap = argparse.ArgumentParser(description="Claude experiment suite runner")
    ap.add_argument("--num", type=int, default=30, help="limit number of experiments")
    ap.add_argument("--out", type=str, default="reports/claude_experiments")
    ap.add_argument("--model", type=str, default="claude-3-5-sonnet")
    ap.add_argument("--dry", action="store_true", help="only write specs JSON; do not call Claude")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.num)
    (outdir / "specs.json").write_text(json.dumps([s.dict() for s in specs], indent=2))

    if args.dry:
        print(f"Wrote {len(specs)} specs to {outdir/'specs.json'}")
        return

    try:
        import anthropic  # noqa
    except Exception as e:
        print("anthropic package not installed. Run: pip install anthropic pydantic")
        raise

    summary: Dict[str, Dict] = {}
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.id} — {spec.title}")
        try:
            res = call_claude(spec, model=args.model)
        except Exception as e:
            res = {"error": str(e)}
        (outdir / f"{spec.id}.json").write_text(json.dumps(res, indent=2))
        # Small delay to be gentle on rate limits
        time.sleep(0.5)
        summary[spec.id] = {"title": spec.title, "status": "ok" if "error" not in res else "error"}

    (outdir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()
