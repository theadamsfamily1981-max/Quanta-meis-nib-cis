#!/usr/bin/env python
"""
Interactive Pareto Dashboard

Streamlit app for visualizing Pareto optimization results.

Usage:
    streamlit run dashboards/pareto_app.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("⚠ Streamlit/Plotly not installed")
    print("Install with: pip install streamlit plotly")
    st = None


def load_dashboard_data(path: str = 'artifacts/pareto/pareto.json'):
    """Load dashboard data from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_pareto_2d(data, obj_x, obj_y):
    """Plot 2D Pareto frontier."""
    if st is None:
        return None

    # Extract points
    all_points = pd.DataFrame(data['all_points'])

    # Separate frontier and non-frontier points
    frontier_points = all_points[all_points['on_frontier']]
    other_points = all_points[~all_points['on_frontier']]

    fig = go.Figure()

    # Plot non-frontier points
    if len(other_points) > 0:
        fig.add_trace(go.Scatter(
            x=[p[obj_x] for p in other_points['objectives']],
            y=[p[obj_y] for p in other_points['objectives']],
            mode='markers',
            name='All Points',
            marker=dict(size=8, color='lightblue', opacity=0.6)
        ))

    # Plot frontier points
    if len(frontier_points) > 0:
        fig.add_trace(go.Scatter(
            x=[p[obj_x] for p in frontier_points['objectives']],
            y=[p[obj_y] for p in frontier_points['objectives']],
            mode='markers',
            name='Pareto Frontier',
            marker=dict(size=12, color='red', symbol='star')
        ))

    fig.update_layout(
        title=f'Pareto Frontier: {obj_x} vs {obj_y}',
        xaxis_title=obj_x,
        yaxis_title=obj_y,
        hovermode='closest'
    )

    return fig


def main():
    """Main dashboard app."""
    if st is None:
        print("Streamlit not available")
        return

    st.set_page_config(page_title="TF-A-N Pareto Dashboard", layout="wide")

    st.title("🎯 TF-A-N Pareto Optimization Dashboard")

    # Sidebar - data loading
    st.sidebar.header("Configuration")

    data_path = st.sidebar.text_input(
        "Dashboard Data Path",
        value="artifacts/pareto/pareto.json"
    )

    if not Path(data_path).exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Run Pareto auto-runner first to generate data")
        return

    # Load data
    data = load_dashboard_data(data_path)

    # Summary metrics
    st.header("Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Runs", data['summary']['num_runs'])

    with col2:
        st.metric("Successful", data['summary']['num_successful'])

    with col3:
        st.metric("Frontier Points", data['summary']['num_frontier'])

    with col4:
        frontier_ratio = data['summary']['frontier_ratio']
        st.metric("Frontier Ratio", f"{frontier_ratio:.1%}")

    # Objectives
    st.header("Objectives")

    minimize = data['objectives']['minimize']
    maximize = data['objectives']['maximize']

    st.write(f"**Minimize:** {', '.join(minimize)}")
    st.write(f"**Maximize:** {', '.join(maximize)}")

    # Pareto frontier visualization
    st.header("Pareto Frontier Visualization")

    all_objectives = minimize + maximize

    if len(all_objectives) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            obj_x = st.selectbox("X-axis", all_objectives, index=0)

        with col2:
            obj_y = st.selectbox("Y-axis", all_objectives, index=1 if len(all_objectives) > 1 else 0)

        # Plot
        fig = plot_pareto_2d(data, obj_x, obj_y)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Frontier table
    st.header("Frontier Points")

    frontier_df = pd.DataFrame(data['frontier'])

    # Expand objectives into columns
    if len(frontier_df) > 0:
        obj_cols = pd.json_normalize(frontier_df['objectives'])
        display_df = pd.concat([
            frontier_df[['config_id']],
            obj_cols
        ], axis=1)

        st.dataframe(display_df, use_container_width=True)

    # Download frontier
    if len(frontier_df) > 0:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Frontier CSV",
            data=csv,
            file_name="pareto_frontier.csv",
            mime="text/csv"
        )


if __name__ == '__main__':
    if st is not None:
        main()
    else:
        print("Run with: streamlit run dashboards/pareto_app.py")
