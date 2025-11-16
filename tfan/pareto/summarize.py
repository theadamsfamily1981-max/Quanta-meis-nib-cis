#!/usr/bin/env python
"""
Pareto Results Summarization and Dashboard Export

Generates:
- Summary statistics
- Pareto frontier visualization data
- Interactive dashboard JSON
- HTML reports

Usage:
    summary = summarize_results(results, frontier)
    export_dashboard(frontier, 'dashboards/pareto_app.json')
"""

import json
import numpy as np
from typing import List, Dict
from pathlib import Path


def summarize_results(
    results: List,
    frontier: List,
    minimize: List[str] = None,
    maximize: List[str] = None
) -> Dict:
    """
    Generate summary statistics.

    Args:
        results: All run results
        frontier: Pareto frontier points
        minimize: Objectives to minimize
        maximize: Objectives to maximize

    Returns:
        Summary dict
    """
    minimize = minimize or []
    maximize = maximize or []

    successful = [r for r in results if r.status == 'success']

    summary = {
        'num_runs': len(results),
        'num_successful': len(successful),
        'num_frontier': len(frontier),
        'frontier_ratio': len(frontier) / len(successful) if successful else 0.0,
        'objectives': minimize + maximize,
        'minimize': minimize,
        'maximize': maximize
    }

    # Objective statistics
    if successful:
        all_objectives = minimize + maximize

        for obj_name in all_objectives:
            values = [r.metrics.get(obj_name, 0) for r in successful]

            summary[f'{obj_name}_mean'] = np.mean(values)
            summary[f'{obj_name}_std'] = np.std(values)
            summary[f'{obj_name}_min'] = np.min(values)
            summary[f'{obj_name}_max'] = np.max(values)

    # Frontier statistics
    if frontier:
        for i, obj_name in enumerate(minimize + maximize):
            frontier_values = [p.objectives[i] for p in frontier]

            summary[f'{obj_name}_frontier_mean'] = np.mean(frontier_values)
            summary[f'{obj_name}_frontier_best'] = np.min(frontier_values)
            summary[f'{obj_name}_frontier_worst'] = np.max(frontier_values)

    # Wall time statistics
    if successful:
        wall_times = [r.wall_time_s for r in successful]

        summary['wall_time_total_s'] = np.sum(wall_times)
        summary['wall_time_mean_s'] = np.mean(wall_times)
        summary['wall_time_max_s'] = np.max(wall_times)

    return summary


def export_dashboard(
    frontier: List,
    results: List,
    output_path: str,
    minimize: List[str] = None,
    maximize: List[str] = None
):
    """
    Export dashboard JSON for visualization.

    Args:
        frontier: Pareto frontier points
        results: All run results
        output_path: Path to save dashboard JSON
        minimize: Objectives to minimize
        maximize: Objectives to maximize
    """
    minimize = minimize or []
    maximize = maximize or []

    # Convert frontier to serializable format
    frontier_data = []
    for point in frontier:
        point_data = {
            'config_id': point.config_id,
            'params': point.params,
            'objectives': {}
        }

        # Reconstruct original objective values
        for i, obj_name in enumerate(minimize):
            point_data['objectives'][obj_name] = float(point.objectives[i])

        for i, obj_name in enumerate(maximize):
            # Negate back to original value
            point_data['objectives'][obj_name] = float(-point.objectives[len(minimize) + i])

        frontier_data.append(point_data)

    # All points data
    all_points_data = []
    successful = [r for r in results if r.status == 'success']

    for result in successful:
        point_data = {
            'config_id': result.config_id,
            'params': result.params,
            'objectives': result.metrics,
            'on_frontier': any(p.config_id == result.config_id for p in frontier)
        }
        all_points_data.append(point_data)

    # Dashboard data
    dashboard = {
        'title': 'TF-A-N Pareto Optimization Dashboard',
        'objectives': {
            'minimize': minimize,
            'maximize': maximize
        },
        'frontier': frontier_data,
        'all_points': all_points_data,
        'summary': summarize_results(results, frontier, minimize, maximize)
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dashboard, f, indent=2)

    print(f"✓ Dashboard exported to {output_path}")


def generate_html_report(
    frontier: List,
    results: List,
    output_path: str,
    minimize: List[str] = None,
    maximize: List[str] = None
):
    """
    Generate HTML report with embedded visualizations.

    Args:
        frontier: Pareto frontier
        results: All results
        output_path: Path to save HTML
        minimize: Objectives to minimize
        maximize: Objectives to maximize
    """
    minimize = minimize or []
    maximize = maximize or []

    summary = summarize_results(results, frontier, minimize, maximize)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TF-A-N Pareto Optimization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .frontier-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        .frontier-table th, .frontier-table td {{ border: 1px solid #ddd; padding: 8px; }}
        .frontier-table th {{ background-color: #4CAF50; color: white; }}
        .frontier-point {{ background-color: #e8f5e9; }}
    </style>
</head>
<body>
    <h1>TF-A-N Pareto Optimization Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric"><strong>Total Runs:</strong> {summary['num_runs']}</div>
        <div class="metric"><strong>Successful:</strong> {summary['num_successful']}</div>
        <div class="metric"><strong>Frontier Points:</strong> {summary['num_frontier']}</div>
        <div class="metric"><strong>Frontier Ratio:</strong> {summary['frontier_ratio']:.2%}</div>
        <div class="metric"><strong>Total Wall Time:</strong> {summary.get('wall_time_total_s', 0) / 3600:.2f} hours</div>
    </div>

    <h2>Objectives</h2>
    <ul>
        <li><strong>Minimize:</strong> {', '.join(minimize)}</li>
        <li><strong>Maximize:</strong> {', '.join(maximize)}</li>
    </ul>

    <h2>Pareto Frontier</h2>
    <table class="frontier-table">
        <tr>
            <th>Config ID</th>
"""

    # Add objective columns
    for obj_name in minimize + maximize:
        html += f"            <th>{obj_name}</th>\n"

    html += "        </tr>\n"

    # Add frontier rows
    for point in frontier:
        html += f"""        <tr class="frontier-point">
            <td>{point.config_id}</td>
"""

        # Add objective values
        for i in range(len(minimize + maximize)):
            value = point.objectives[i]

            # Negate maximize objectives back
            if i >= len(minimize):
                value = -value

            html += f"            <td>{value:.4f}</td>\n"

        html += "        </tr>\n"

    html += """    </table>
</body>
</html>
"""

    # Save HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ HTML report generated: {output_path}")
