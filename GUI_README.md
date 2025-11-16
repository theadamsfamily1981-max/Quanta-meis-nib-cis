# 🧠 TF-A-N Control Center - User Guide

**Easy-to-use graphical interface for testing, benchmarking, and exploring all TF-A-N subsystems.**

No command-line experience needed! Just point and click.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

Open a terminal in your repository directory and run:

```bash
pip install streamlit plotly pandas torch numpy
```

### Step 2: Launch the GUI

**On Linux/Mac:**
```bash
./launch_gui.sh
```

**On Windows:**
```
launch_gui.bat
```

**Or manually:**
```bash
streamlit run tfan_gui.py
```

### Step 3: Open Your Browser

The GUI will automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, just copy that URL into your browser.

---

## 📖 User Guide

### 🏠 Home Page

The landing page shows:
- **Available Subsystems**: All implemented TF-A-N components with their hard gates
- **System Info**: Python/PyTorch/CUDA status
- **Dependency Check**: Verify all required packages are installed

**What to do:**
1. Set your repository directory in the sidebar (usually auto-detected)
2. Check that dependencies are installed
3. Navigate to other pages using the sidebar

---

### 🧪 Run Tests Page

**Purpose**: Run comprehensive test suites to verify everything works.

**How to use:**

1. **Select Test Suites**: Check the boxes for which subsystems you want to test
   - Imports: Quick import checks
   - PGU TurboCache: Proof cache tests
   - Pareto Auto-Runner: Multi-objective optimization tests
   - Multimodal Fusion: Audio/video/PAD tests
   - eBPF Telemetry: Tracing tests (Linux only)
   - Fused Radial Attention: CUDA kernel tests
   - FSDP Orchestrator: Distributed training tests
   - HyperKG: Knowledge graph tests

2. **Set Options**:
   - Verbose output: See detailed logs
   - Save results: Export to JSON file

3. **Click "Run Selected Tests"**: Tests will run automatically

4. **View Results**: See pass/fail status, errors, and detailed output

**Tips:**
- Start with all tests checked to verify your installation
- Uncheck eBPF tests if you're not on Linux
- Use verbose mode to debug failures

---

### 📊 Benchmarks Page

**Purpose**: Measure performance of each subsystem.

**Available Benchmarks:**

#### 1. PGU TurboCache - Cache Performance
- **What it does**: Tests cache hit rate and latency
- **Settings**:
  - Corpus size: Number of unique formulas (100-10,000)
  - Iterations: How many queries to run (10-1,000)
- **What to look for**: ≥60% hit rate, p95 ≤120ms

#### 2. Fused Radial Attention - CUDA Kernel
- **What it does**: Compares fused vs unfused attention
- **Settings**:
  - Batch size: 1-16
  - Sequence length: 128-2048
  - Iterations: 10-200
  - Sweep mode: Test multiple sequence lengths
- **What to look for**: ≥2× speedup, <1e-3 error

#### 3. FSDP Orchestrator - Multi-GPU Training
- **What it does**: Compares FSDP vs DDP
- **Requirements**: 4 GPUs (will fall back to single GPU otherwise)
- **What to look for**: ≥1.6× speedup on 4 GPUs

#### 4. HyperKG - Hyperbolic Embeddings
- **What it does**: Tests knowledge graph embeddings
- **Settings**:
  - Dataset: synthetic, family, wordnet
  - Embedding dim: 16-256
  - Training epochs: 50-500
- **What to look for**: MRR ≥ baseline + 5%

**How to use:**
1. Select a benchmark from the dropdown
2. Adjust settings (or use defaults)
3. Click "Run [Benchmark Name]"
4. Results appear below with timing and metrics

---

### 🎮 Demos Page

**Purpose**: Interactive demonstrations of key features.

**Available Demos:**

#### 1. HyperKG + Datalog - Family Relationships
- Creates a family tree knowledge graph
- Adds Datalog rules (ancestor, sibling)
- Trains hyperbolic embeddings
- Queries with PGU verification
- **Duration**: ~2 minutes

#### 2. FSDP Training - Simple Transformer
- Trains a small transformer model
- Compares single GPU vs FSDP
- Shows memory and speed improvements
- **Duration**: ~3-5 minutes

#### 3. Pareto Optimization - Multi-objective Search
- Searches for Pareto-optimal configurations
- Visualizes frontier
- Shows EHVI computation
- **Duration**: ~1 minute

**How to use:**
1. Select a demo
2. Adjust parameters (optional)
3. Click "Run [Demo Name]"
4. Watch the output in real-time

---

### 📈 Results Page

**Purpose**: View and analyze test results.

**Features:**

1. **Summary Metrics**:
   - Total tests run
   - Passed/Failed/Errors
   - Overall pass rate
   - Total duration

2. **Pass Rate Gauge**:
   - Visual indicator of overall health
   - Green = good (>80%)
   - Yellow = warning (50-80%)
   - Red = critical (<50%)

3. **Results by Subsystem**:
   - Bar chart showing tests passed per subsystem
   - Table with pass rates and durations

4. **Detailed Results**:
   - Filter by status (pass/fail/error)
   - View individual test outputs
   - Inspect error messages

5. **Failed Tests Section**:
   - Expandable details for each failure
   - Full error messages
   - Output logs

**How to use:**
1. Run tests from the "Run Tests" page first
2. Navigate to this page to see results
3. Use filters to focus on specific issues
4. Export results to JSON for sharing

---

## 🎯 Common Workflows

### First-Time Setup Verification

1. Launch GUI
2. Go to "Run Tests" page
3. Check all test suites
4. Click "Run Selected Tests"
5. Wait for completion (5-10 minutes)
6. Go to "Results" page
7. Verify >90% pass rate

If any tests fail:
- Check error messages in Results page
- Ensure all dependencies are installed
- Verify you're in the correct directory

---

### Before Committing Code

1. Run tests for the subsystem you changed
2. Run relevant benchmark to check performance
3. Verify hard gates are still met
4. Check that pass rate didn't decrease

---

### Debugging a Subsystem

1. Go to "Run Tests" page
2. Select only the failing subsystem
3. Enable "Verbose output"
4. Run tests
5. Check detailed output in Results page
6. Use error messages to identify issues

---

### Performance Tuning

1. Go to "Benchmarks" page
2. Run baseline benchmark with default settings
3. Modify code
4. Re-run benchmark
5. Compare results to verify improvement
6. Repeat until hard gates are met

---

## ⚙️ Settings & Configuration

### Sidebar Settings

**Repository Directory**:
- Auto-detected by default
- Change if running GUI from different location
- Must contain `tfan/` directory
- Shows green checkmark if valid

**Navigation**:
- Radio buttons to switch pages
- Current page is highlighted

**System Info**:
- Python version
- PyTorch version
- CUDA availability
- GPU count (if CUDA available)

---

## 🐛 Troubleshooting

### "Invalid repository" error

**Problem**: Sidebar shows "✗ Invalid repository"

**Solution**:
1. Click in the "Repository Directory" field
2. Enter the full path to your cloned repo
3. Make sure the path contains a `tfan/` folder

---

### GUI won't start

**Problem**: `streamlit run tfan_gui.py` fails

**Solution**:
```bash
# Install/reinstall Streamlit
pip install --upgrade streamlit plotly pandas

# Try running directly
python -m streamlit run tfan_gui.py
```

---

### Tests are failing

**Problem**: Many tests show "✗ Failed" status

**Solution**:
1. Check "System Info" in sidebar
2. Ensure Python ≥3.8, PyTorch installed
3. Run with verbose output enabled
4. Check error messages in Results page
5. Install missing dependencies:
   ```bash
   pip install torch numpy pandas plotly streamlit
   ```

---

### Benchmarks are slow

**Problem**: Benchmarks take a very long time

**Solution**:
- Reduce iterations in settings
- Use smaller batch sizes
- Reduce sequence length
- Disable sweep mode
- Run on GPU if available

---

### "Module not found" errors

**Problem**: Tests fail with import errors

**Solution**:
```bash
# Ensure you're in the repo directory
cd /path/to/Quanta-meis-nib-cis

# Install all dependencies
pip install -e .

# Or manually
pip install torch numpy pandas plotly streamlit
```

---

## 📊 Understanding Results

### Test Status Icons

- ✓ **Pass**: Test succeeded, all checks passed
- ✗ **Fail**: Test failed, but ran to completion
- ⚠ **Error**: Test crashed or timed out
- ⊘ **Skip**: Test was skipped (e.g., no GPU)

### Hard Gates

Each subsystem has **hard gates** - performance/accuracy thresholds that must be met:

| Subsystem | Hard Gates |
|-----------|-----------|
| PGU | ≥60% hit rate, p95 ≤120ms |
| Pareto | ≥6 frontier points, ≤6h |
| Multimodal | PAD MAE ≤0.15 |
| eBPF | ≥95% attribution, ≤3s alert |
| Fused Attention | ≥2× speedup, <1e-3 error |
| FSDP | ≥1.6× speedup vs DDP |
| HyperKG | MRR ≥ Euclidean + 5%, PGU ≥95% |

If benchmarks show gate violations, the implementation may need tuning.

### Pass Rate Targets

- **≥95%**: Excellent - production ready
- **90-95%**: Good - minor issues only
- **80-90%**: Fair - needs attention
- **<80%**: Critical - major issues

---

## 🔧 Advanced Usage

### Running from Command Line

If you prefer CLI:

```bash
# Run all tests
python tests/run_all_tests.py

# Run with verbose output
python tests/run_all_tests.py --verbose

# Save results to JSON
python tests/run_all_tests.py --output results.json

# Run specific benchmark
python benchmarks/bench_hyperkg.py --dataset synthetic --epochs 100

# Run demo
python examples/hyperkg_datalog_demo.py
```

### Customizing the GUI

The GUI is built with Streamlit. To modify:

1. Edit `tfan_gui.py`
2. Add new pages/sections using Streamlit components
3. Reload browser to see changes (auto-reloads on save)

### CI/CD Integration

Use the CLI test runner in GitHub Actions:

```yaml
- name: Run TF-A-N tests
  run: python tests/run_all_tests.py --output test_results.json

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_results.json
```

---

## 📝 Tips & Best Practices

1. **Run tests before and after changes**: Catch regressions early

2. **Use verbose mode for debugging**: Get detailed error messages

3. **Save results to JSON**: Compare across runs, track improvements

4. **Run benchmarks on GPU**: Get accurate performance numbers

5. **Check system info**: Ensure CUDA is detected if you have a GPU

6. **Filter results by subsystem**: Focus on what you're working on

7. **Export and share results**: Send JSON files to collaborators

8. **Run demos to learn**: See how each component works in practice

---

## 🆘 Getting Help

If you encounter issues:

1. Check the error message in verbose output
2. Verify dependencies are installed
3. Ensure repository path is correct
4. Check system requirements (Python ≥3.8, etc.)
5. Review the error in the Results page

For bugs or feature requests:
- Open an issue on GitHub
- Include error messages and logs
- Attach test results JSON if available

---

## 📚 Next Steps

After verifying everything works:

1. **Explore Demos**: Learn how each component works
2. **Run Benchmarks**: Measure baseline performance
3. **Read Subsystem READMEs**: Deep dive into specific components
4. **Modify Code**: Make changes and verify with tests
5. **Share Results**: Export JSON and collaborate

---

**Happy testing! 🚀**
