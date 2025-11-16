#!/usr/bin/env python
"""
TF-A-N Control Center - Interactive GUI

Easy-to-use graphical interface for running tests, benchmarks, and demos
across all TF-A-N subsystems.

Usage:
    streamlit run tfan_gui.py

    # Or with specific port
    streamlit run tfan_gui.py --server.port 8501
"""

import streamlit as st
import subprocess
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="TF-A-N Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'repo_dir' not in st.session_state:
    st.session_state.repo_dir = str(Path.cwd())
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'running' not in st.session_state:
    st.session_state.running = False


def run_command(command, cwd=None, timeout=300):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd or st.session_state.repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=isinstance(command, str)
        )
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': f'Timeout after {timeout}s'
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }


def check_dependencies():
    """Check if required dependencies are installed."""
    deps = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'streamlit': 'Streamlit'
    }

    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)

    return missing


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🧠 TF-A-N Control Center")
st.sidebar.markdown("---")

# Repository directory
st.sidebar.subheader("📁 Repository")
repo_dir = st.sidebar.text_input(
    "Repository Directory",
    value=st.session_state.repo_dir,
    help="Path to your cloned TF-A-N repository"
)

if repo_dir != st.session_state.repo_dir:
    st.session_state.repo_dir = repo_dir
    st.rerun()

# Check if valid repo
repo_path = Path(st.session_state.repo_dir)
is_valid_repo = (repo_path / 'tfan').exists()

if is_valid_repo:
    st.sidebar.success("✓ Valid repository")
else:
    st.sidebar.error("✗ Invalid repository (tfan/ not found)")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧪 Run Tests", "📊 Benchmarks", "🎮 Demos", "📈 Results"]
)

st.sidebar.markdown("---")

# System info
st.sidebar.subheader("System Info")
import torch
st.sidebar.text(f"Python: {sys.version.split()[0]}")
st.sidebar.text(f"PyTorch: {torch.__version__}")
st.sidebar.text(f"CUDA: {'✓' if torch.cuda.is_available() else '✗'}")
if torch.cuda.is_available():
    st.sidebar.text(f"GPUs: {torch.cuda.device_count()}")


# ============================================================================
# MAIN CONTENT
# ============================================================================

if page == "🏠 Home":
    st.title("🧠 TF-A-N Control Center")
    st.markdown("### Topological Foundations for Artificial Networks")

    st.markdown("""
    Welcome to the TF-A-N Control Center! This interactive GUI lets you:

    - **🧪 Run Tests**: Execute comprehensive test suites with one click
    - **📊 Benchmarks**: Run performance benchmarks for all subsystems
    - **🎮 Demos**: Interactive demonstrations of each component
    - **📈 Results**: View and analyze test results

    ## 🚀 Quick Start

    1. **Set Repository Path**: Enter the path to your cloned repository in the sidebar
    2. **Choose a Page**: Select what you want to do from the sidebar
    3. **Click and Go**: All operations are point-and-click!

    ## 📦 Available Subsystems

    """)

    subsystems = [
        {
            'name': 'PGU TurboCache',
            'issue': '#4',
            'description': 'Alpha-renaming proof cache with SQLite/LMDB backends',
            'gates': '≥60% hit rate, p95 ≤120ms'
        },
        {
            'name': 'Pareto Auto-Runner',
            'issue': '#15',
            'description': 'Multi-objective optimization with EHVI',
            'gates': '≥6 frontier points, ≤6h wall-time'
        },
        {
            'name': 'Multimodal Fusion + PAD',
            'issue': '#11/12',
            'description': 'Emotion-modulated multimodal fusion',
            'gates': 'PAD MAE ≤0.15, TTW drift <5%'
        },
        {
            'name': 'eBPF + GPU Telemetry',
            'issue': '#16',
            'description': 'Kernel-level tracing with Grafana dashboards',
            'gates': '≥95% attribution, ≤3s alert latency'
        },
        {
            'name': 'Fused Radial Attention',
            'issue': '#18',
            'description': 'CUDA kernel for sparse attention',
            'gates': '≥2× speedup, <1e-3 error, -20% VRAM'
        },
        {
            'name': 'FSDP Orchestrator',
            'issue': '#19',
            'description': 'Multi-GPU training with ZeRO-3',
            'gates': '≥1.6× speedup vs DDP (4 GPUs)'
        },
        {
            'name': 'HyperKG + Datalog',
            'issue': '#17',
            'description': 'Hyperbolic KG with symbolic verification',
            'gates': 'MRR ≥ Euclidean + 5%, PGU agreement ≥95%'
        }
    ]

    df = pd.DataFrame(subsystems)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Check dependencies
    st.markdown("---")
    st.subheader("🔍 Dependency Check")

    missing = check_dependencies()
    if not missing:
        st.success("✓ All required dependencies installed")
    else:
        st.warning(f"⚠ Missing dependencies: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}")


elif page == "🧪 Run Tests":
    st.title("🧪 Run Tests")

    if not is_valid_repo:
        st.error("Please set a valid repository directory in the sidebar")
        st.stop()

    st.markdown("""
    Select which test suites to run. Tests will execute sequentially and results
    will be displayed below.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test Suites")

        test_suites = {
            'imports': st.checkbox("Import Tests", value=True),
            'pgu': st.checkbox("PGU TurboCache", value=True),
            'pareto': st.checkbox("Pareto Auto-Runner", value=True),
            'multimodal': st.checkbox("Multimodal Fusion", value=True),
            'ebpf': st.checkbox("eBPF Telemetry", value=False, help="Requires Linux"),
            'fused_attention': st.checkbox("Fused Radial Attention", value=True),
            'fsdp': st.checkbox("FSDP Orchestrator", value=True),
            'hyperkg': st.checkbox("HyperKG", value=True),
        }

    with col2:
        st.subheader("Options")

        verbose = st.checkbox("Verbose output", value=False)
        save_results = st.checkbox("Save results to JSON", value=True)

        if save_results:
            output_file = st.text_input(
                "Output file",
                value=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

    st.markdown("---")

    # Run button
    if st.button("▶️ Run Selected Tests", type="primary", use_container_width=True):
        st.session_state.running = True

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Build command
        cmd = [
            sys.executable,
            "tests/run_all_tests.py",
            "--repo-dir", st.session_state.repo_dir
        ]

        if verbose:
            cmd.append("--verbose")

        if save_results:
            cmd.extend(["--output", output_file])

        # Run tests
        status_text.text("Running tests...")
        result = run_command(cmd, timeout=600)

        progress_bar.progress(100)
        st.session_state.running = False

        # Display results
        if result['success']:
            st.success("✓ All tests completed!")

            if save_results and Path(output_file).exists():
                with open(output_file, 'r') as f:
                    st.session_state.test_results = json.load(f)

                # Display summary
                st.subheader("Test Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total", st.session_state.test_results['total'])
                with col2:
                    st.metric("✓ Passed", st.session_state.test_results['passed'])
                with col3:
                    st.metric("✗ Failed", st.session_state.test_results['failed'])
                with col4:
                    st.metric("Pass Rate", f"{st.session_state.test_results['pass_rate']:.1%}")

                # Results table
                if st.session_state.test_results['results']:
                    df = pd.DataFrame(st.session_state.test_results['results'])
                    st.dataframe(df, use_container_width=True)

        else:
            st.error("✗ Tests failed")
            with st.expander("Error details"):
                st.code(result['stderr'])

        # Show output
        with st.expander("Full output"):
            st.code(result['stdout'])


elif page == "📊 Benchmarks":
    st.title("📊 Benchmarks")

    if not is_valid_repo:
        st.error("Please set a valid repository directory in the sidebar")
        st.stop()

    st.markdown("Run performance benchmarks for each subsystem.")

    benchmark_selection = st.selectbox(
        "Select Benchmark",
        [
            "PGU TurboCache - Cache Performance",
            "Pareto Auto-Runner - Multi-objective Optimization",
            "Fused Radial Attention - CUDA Kernel vs Unfused",
            "FSDP Orchestrator - FSDP vs DDP (requires 4 GPUs)",
            "HyperKG - Hyperbolic vs Euclidean Embeddings"
        ]
    )

    st.markdown("---")

    if "PGU" in benchmark_selection:
        st.subheader("PGU TurboCache Benchmark")

        col1, col2 = st.columns(2)
        with col1:
            corpus_size = st.number_input("Corpus size", min_value=10, max_value=10000, value=1000)
        with col2:
            iterations = st.number_input("Iterations", min_value=10, max_value=1000, value=100)

        if st.button("Run PGU Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                cmd = [
                    sys.executable,
                    "tfan/pgu/corpus_replay.py",
                    "--size", str(corpus_size),
                    "--iterations", str(iterations)
                ]

                result = run_command(cmd)

                if result['success']:
                    st.success("✓ Benchmark completed")
                    st.code(result['stdout'])
                else:
                    st.error("✗ Benchmark failed")
                    st.code(result['stderr'])

    elif "Fused Radial" in benchmark_selection:
        st.subheader("Fused Radial Attention Benchmark")

        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=4)
        with col2:
            seq_len = st.number_input("Sequence length", min_value=128, max_value=2048, value=512, step=128)
        with col3:
            iterations = st.number_input("Iterations", min_value=10, max_value=200, value=100)

        sweep_mode = st.checkbox("Sweep sequence lengths")

        if st.button("Run Fused Attention Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                cmd = [
                    sys.executable,
                    "benchmarks/bench_fused_radial.py",
                    "--batch-size", str(batch_size),
                    "--seq-len", str(seq_len),
                    "--iterations", str(iterations)
                ]

                if sweep_mode:
                    cmd.append("--sweep-seq-len")

                result = run_command(cmd, timeout=600)

                if result['success']:
                    st.success("✓ Benchmark completed")
                    st.code(result['stdout'])
                else:
                    st.error("✗ Benchmark failed")
                    st.code(result['stderr'])

    elif "HyperKG" in benchmark_selection:
        st.subheader("HyperKG Benchmark")

        col1, col2, col3 = st.columns(3)
        with col1:
            dataset = st.selectbox("Dataset", ["synthetic", "family", "wordnet"])
        with col2:
            dims = st.number_input("Embedding dim", min_value=16, max_value=256, value=64, step=16)
        with col3:
            epochs = st.number_input("Training epochs", min_value=50, max_value=500, value=200)

        if dataset == "synthetic":
            col1, col2 = st.columns(2)
            with col1:
                depth = st.number_input("Tree depth", min_value=2, max_value=6, value=4)
            with col2:
                branching = st.number_input("Branching factor", min_value=2, max_value=5, value=3)

        if st.button("Run HyperKG Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                cmd = [
                    sys.executable,
                    "benchmarks/bench_hyperkg.py",
                    "--dataset", dataset,
                    "--dims", str(dims),
                    "--epochs", str(epochs)
                ]

                if dataset == "synthetic":
                    cmd.extend(["--depth", str(depth), "--branching", str(branching)])

                result = run_command(cmd, timeout=600)

                if result['success']:
                    st.success("✓ Benchmark completed")
                    st.code(result['stdout'])
                else:
                    st.error("✗ Benchmark failed")
                    st.code(result['stderr'])


elif page == "🎮 Demos":
    st.title("🎮 Interactive Demos")

    if not is_valid_repo:
        st.error("Please set a valid repository directory in the sidebar")
        st.stop()

    st.markdown("Run interactive demonstrations of each subsystem.")

    demo_selection = st.selectbox(
        "Select Demo",
        [
            "HyperKG + Datalog - Family Relationships",
            "FSDP Training - Simple Transformer",
            "Pareto Optimization - Multi-objective Search"
        ]
    )

    st.markdown("---")

    if "HyperKG" in demo_selection:
        st.subheader("HyperKG + Datalog Demo")
        st.markdown("""
        This demo shows:
        - Creating a family relationship KG
        - Adding Datalog rules (transitivity, symmetry)
        - Training hyperbolic embeddings
        - Querying with PGU verification
        """)

        if st.button("Run HyperKG Demo", type="primary"):
            with st.spinner("Running demo..."):
                cmd = [sys.executable, "examples/hyperkg_datalog_demo.py"]
                result = run_command(cmd, timeout=300)

                if result['success']:
                    st.success("✓ Demo completed")
                    st.code(result['stdout'])
                else:
                    st.error("✗ Demo failed")
                    st.code(result['stderr'])

    elif "FSDP" in demo_selection:
        st.subheader("FSDP Training Demo")

        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
            use_fsdp = st.checkbox("Use FSDP", value=False, help="Requires multiple GPUs")
        with col2:
            batch_size = st.number_input("Batch size", min_value=1, max_value=8, value=4)
            cpu_offload = st.checkbox("CPU offload", value=False)

        if st.button("Run FSDP Demo", type="primary"):
            with st.spinner("Running demo..."):
                cmd = [
                    sys.executable,
                    "examples/train_with_fsdp.py",
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size)
                ]

                if use_fsdp:
                    cmd.append("--use-fsdp")
                if cpu_offload:
                    cmd.append("--cpu-offload")

                result = run_command(cmd, timeout=600)

                if result['success']:
                    st.success("✓ Demo completed")
                    st.code(result['stdout'])
                else:
                    st.error("✗ Demo failed")
                    st.code(result['stderr'])


elif page == "📈 Results":
    st.title("📈 Test Results")

    if st.session_state.test_results is None:
        st.info("No test results available. Run tests from the 'Run Tests' page first.")
        st.stop()

    results = st.session_state.test_results

    # Summary metrics
    st.subheader("Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Tests", results['total'])
    with col2:
        st.metric("✓ Passed", results['passed'], delta=None)
    with col3:
        st.metric("✗ Failed", results['failed'], delta=None)
    with col4:
        st.metric("⚠ Errors", results['errors'], delta=None)
    with col5:
        st.metric("Duration", f"{results['duration_s']:.1f}s")

    # Pass rate gauge
    st.subheader("Pass Rate")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=results['pass_rate'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pass Rate (%)"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Results by subsystem
    st.subheader("Results by Subsystem")

    if results['results']:
        df = pd.DataFrame(results['results'])

        # Group by subsystem
        subsystem_stats = df.groupby('subsystem').agg({
            'status': lambda x: (x == 'pass').sum(),
            'duration_s': 'sum'
        }).reset_index()
        subsystem_stats.columns = ['Subsystem', 'Passed', 'Duration (s)']

        # Add total tests per subsystem
        subsystem_total = df.groupby('subsystem').size().reset_index(name='Total')
        subsystem_stats = subsystem_stats.merge(subsystem_total, on='Subsystem')
        subsystem_stats['Pass Rate'] = subsystem_stats['Passed'] / subsystem_stats['Total']

        st.dataframe(subsystem_stats, use_container_width=True, hide_index=True)

        # Chart
        fig = px.bar(
            subsystem_stats,
            x='Subsystem',
            y='Passed',
            color='Pass Rate',
            color_continuous_scale='RdYlGn',
            title='Tests Passed by Subsystem'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed results
        st.subheader("Detailed Results")

        # Filter
        status_filter = st.multiselect(
            "Filter by status",
            options=['pass', 'fail', 'error', 'skip'],
            default=['pass', 'fail', 'error']
        )

        filtered_df = df[df['status'].isin(status_filter)]
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Failed tests details
        if results['failed'] > 0 or results['errors'] > 0:
            st.subheader("❌ Failed Tests")

            failed_df = df[df['status'].isin(['fail', 'error'])]

            for _, row in failed_df.iterrows():
                with st.expander(f"{row['subsystem']}: {row['name']}"):
                    st.write(f"**Status:** {row['status']}")
                    st.write(f"**Duration:** {row['duration_s']:.2f}s")

                    if row['error']:
                        st.write("**Error:**")
                        st.code(row['error'])

                    if row['output']:
                        st.write("**Output:**")
                        st.code(row['output'])


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
TF-A-N Control Center | Topological Foundations for Artificial Networks
</div>
""", unsafe_allow_html=True)
