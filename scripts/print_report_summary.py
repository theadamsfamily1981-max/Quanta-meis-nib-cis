import json
from pathlib import Path

if __name__ == "__main__":
    report = Path(__file__).parent.parent / "reports" / "report_2025-10-31.json"
    data = json.loads(report.read_text())
    lines = []
    lines.append(f"# {data['report_date']} Validation Summary")
    lines.append("")
    lines.append(f"System: **{data['system_name']}**")
    gates = data['deployment_status']['all_critical_gates']
    lines.append(f"Critical Gates: **{gates}**")
    lines.append("")
    lines.append(f"Accuracy: {data['metrics_comparison']['accuracy']['current']}")
    lines.append(f"OOD AUROC: {data['metrics_comparison']['ood_auroc']['current']}")
    print("\n".join(lines))
