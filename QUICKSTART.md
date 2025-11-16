# 🚀 TF-A-N Quick Start Guide

**Get up and running in 5 minutes!**

---

## ⚡ Fastest Path (GUI)

### 1. Install Dependencies
```bash
pip install streamlit plotly pandas torch numpy
```

### 2. Launch GUI
```bash
./launch_gui.sh   # Linux/Mac
# OR
launch_gui.bat    # Windows
```

### 3. Run Tests
- Click "🧪 Run Tests" in sidebar
- Check all boxes
- Click "▶️ Run Selected Tests"
- Wait ~5 minutes for verification

**Done!** Your browser shows results at `http://localhost:8501`

---

## 🔧 Manual Setup (CLI)

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd Quanta-meis-nib-cis
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests
```bash
python tests/run_all_tests.py --verbose
```

### 3. Run a Demo
```bash
python examples/hyperkg_datalog_demo.py
```

---

## 📦 What's Included

### 8 High-Impact Subsystems

1. **PGU TurboCache** (#4) - Proof caching with ≥60% hit rate
2. **AEPO Agent** (#5) - Tool-use optimization (partial)
3. **Pareto Auto-Runner** (#15) - Multi-objective optimization
4. **Multimodal Fusion + PAD** (#11/#12) - Emotion-modulated fusion
5. **eBPF + GPU Telemetry** (#16) - Kernel-level tracing
6. **Fused Radial Attention** (#18) - CUDA kernel with 2× speedup
7. **FSDP Orchestrator** (#19) - Multi-GPU training with ZeRO-3
8. **HyperKG + Datalog** (#17) - Hyperbolic KG + symbolic reasoning

### Interactive GUI

- 🏠 Home: System overview
- 🧪 Tests: Run test suites
- 📊 Benchmarks: Performance testing
- 🎮 Demos: Interactive demos
- 📈 Results: Visual analytics

---

## 🎯 Try These First

### Test Everything
```bash
./launch_gui.sh
# Click: Run Tests → Check All → Run
```

### Benchmark HyperKG
```bash
python benchmarks/bench_hyperkg.py --dataset synthetic --epochs 100
```

### Run Family KG Demo
```bash
python examples/hyperkg_datalog_demo.py
```

### Train with FSDP
```bash
python examples/train_with_fsdp.py --epochs 3
```

---

## 📚 Next Steps

1. **Explore Demos**: See `examples/` folder
2. **Read Subsystem Docs**: Each has a README in its directory
3. **Run Benchmarks**: Compare performance
4. **Check Hard Gates**: Verify thresholds are met

### Key Documentation

- `GUI_README.md` - Complete GUI guide
- `tfan/pgu/README.md` - PGU TurboCache
- `tfan/kg/README.md` - HyperKG + Datalog
- `tfan/distributed/README.md` - FSDP Orchestrator
- `kernels/ssa/README.md` - Fused Attention

---

## 🆘 Troubleshooting

### GUI won't start
```bash
pip install --upgrade streamlit plotly pandas
python -m streamlit run tfan_gui.py
```

### Tests failing
```bash
# Verify installation
pip install -r requirements.txt
pip install -e .

# Run with verbose
python tests/run_all_tests.py --verbose
```

### Import errors
```bash
# Ensure you're in repo directory
cd /path/to/Quanta-meis-nib-cis

# Reinstall package
pip install -e .
```

---

## ✅ Verification Checklist

- [ ] GUI launches successfully
- [ ] All tests pass (>90% pass rate)
- [ ] At least one demo runs
- [ ] CUDA detected (if you have GPU)
- [ ] Hard gates met in benchmarks

---

**Ready to dive deeper? Check the full documentation in each subsystem's README!** 🚀
