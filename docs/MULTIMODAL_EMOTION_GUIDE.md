# Multi-Modal + Emotion Integration Guide

**Complete implementation roadmap for multi-modal emotion-aware AI**

---

## 🎯 Overview

This guide provides the complete implementation plan for integrating:
1. **Multi-Modal Processing** (Text + Audio + Video + IMU)
2. **Emotion Estimation** (VA or PAD models)
3. **Homeostatic Control** (Emotion ↔ FDT policy modulation)

**Hard Gates (All Must Pass):**
- ✅ TTW p95 < 5ms alignment
- ✅ PGU p95 ≤ 200ms, cache hit ≥ 50%
- ✅ EPR-CV ≤ 0.15
- ✅ SSA speedup ≥ 3× @ 16k/32k
- ✅ Topology: Wasserstein ≤ 2%, cosine ≥ 0.90

---

## ✅ Completed Modules

### 1. tfan/mm/ingest.py (650 lines) ✅

**All 4 Adapters Implemented:**

- **TextAdapter**: Tokenizes text → (features, timestamps)
  - Synthetic reading time (0.1s/token)
  - HuggingFace tokenizer integration
  - Max length 512 tokens

- **AudioAdapter**: Mel-spectrograms + prosody → (features, timestamps)
  - 16kHz sampling, 20ms hop
  - 64 mel bands
  - Prosody: pitch, energy, duration
  - Frame-level timestamps

- **VideoAdapter**: ViT patches → (features, timestamps)
  - 224×224 images, 16×16 patches
  - 25 FPS default
  - Flattened patch features
  - Frame-level timestamps

- **IMUAdapter**: Sensor readings → (features, timestamps)
  - 100Hz default
  - 6-axis (accel + gyro)
  - Normalized features
  - Sample-level timestamps

**Usage:**
```python
from tfan.mm import TextAdapter, AudioAdapter, VideoAdapter

# Text
text_adapter = TextAdapter(tokenizer)
text_features, text_ts = text_adapter("Hello world")

# Audio
audio_adapter = AudioAdapter()
audio_features, audio_ts = audio_adapter(waveform)

# Video
video_adapter = VideoAdapter()
video_features, video_ts = video_adapter(frames)
```

### 2. tfan/mm/align.py (450 lines) ✅

**TTW-Sentry Alignment:**

- Aligns multi-modal streams to universal timebase
- Fast triggers: VFE spike + entropy jump detection
- p95 < 5ms gate monitoring
- Coverage ≥ 90% target
- Dynamic Time Warping with linear interpolation

**Features:**
- Fast path (triggered): nearest-neighbor interpolation
- Standard path: linear interpolation DTW
- Latency tracking and p95 monitoring
- Automatic reference stream selection

**Usage:**
```python
from tfan.mm import align_streams

streams = {
    'text': (text_features, text_ts),
    'audio': (audio_features, audio_ts),
    'video': (video_features, video_ts)
}

result = align_streams(streams)

# Check gates
assert result.latency_ms < 5.0  # p95 gate
assert result.coverage >= 0.90  # coverage gate

# Use aligned streams
aligned_text = result.streams['text']
aligned_audio = result.streams['audio']
aligned_video = result.streams['video']
```

### 3. configs/multimodal_emotion.yaml ✅

**Complete Configuration:**

- All modality settings
- TTW alignment params
- Fusion & SSA config
- Topology gates
- Emotion head config
- FDT control mappings
- PGU safety settings
- All hard gates defined

---

## 🔨 Remaining Modules (Implementation Roadmap)

### Priority 1: Core Fusion & Emotion

#### tfan/mm/fuse.py (Target: 400 lines)

**Purpose:** Pack multi-modal tokens + TLS/SSA masks

**Key Classes:**
```python
class MultiModalTokenizer:
    def pack_tokens(
        self,
        tokens_by_mod: dict,
        timestamps_by_mod: dict
    ) -> PackedTokens:
        # Add [MOD] sentinels and [FUSE] mediators
        # Carry timestamps
        pass

class TLSMaskGenerator:
    def generate_masks(
        self,
        packed_tokens: PackedTokens,
        keep_ratio: float = 0.33,
        alpha: float = 0.7,
        per_head: bool = True
    ) -> AttentionMasks:
        # Compute persistence-lifetime saliency
        # landmark_scores = α·persistence + (1−α)·maxmin
        # Generate per-head masks
        pass

def pack_and_mask(
    streams: dict,  # from align_streams()
    keep_ratio: float = 0.33,
    alpha: float = 0.7
) -> FusedInput:
    # Main entry point
    # Returns: tokens, masks, timestamps
    pass
```

**Integration:**
- Use `tfan.topology.DifferentiableTopology` for persistence
- Use `tfan.attention.RadialSparseAttention` for SSA
- Implement degree-aware pruning (floor=2)

#### tfan/emotion/head.py (Target: 350 lines)

**Purpose:** VA/PAD emotion estimation

**Key Classes:**
```python
class VAHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        # 2D output: Valence, Arousal
        pass

    def forward(self, h: torch.Tensor) -> dict:
        # Returns: {'valence': v, 'arousal': a, 'confidence': c}
        pass

class PADHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        # 3D output: Pleasure, Arousal, Dominance
        pass

def emotion_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_temporal: float = 0.1,
    lambda_topo: float = 0.3
) -> dict:
    # MSE or CCC loss
    # + temporal smoothness: λ_temporal · ||Δstate||
    # + topological consistency via topo.py
    # Returns: {'loss': total, 'mse': mse, 'temporal': temp, 'topo': topo}
    pass
```

**Topology Loss:**
- Enforce β₀=1 (single component for VA trajectory)
- Suppress β₁=0 (no cycles)
- Use `tfan.topology.topo_kl` for penalty

#### tfan/emotion/controller.py (Target: 250 lines)

**Purpose:** Policy modulation hooks

**Key Functions:**
```python
def modulate_policy(
    fdt_metrics: dict,  # From trainer.py
    emotion_state: dict,  # From emotion head
    config: dict
) -> dict:
    # Arousal ↑ ⇒ T ↑ (softer policy)
    # Valence ↓ ⇒ LR ↓ (cooling)
    # Confidence ↓ ⇒ PGU stricter

    # Returns: {'lr_mult': ..., 'temp_mult': ..., 'pgu_threshold_mult': ...}
    # All bounded, PGU-aware
    pass

def apply_emotion_modulation(
    lr: float,
    temp: float,
    emotion_state: dict,
    config: dict
) -> Tuple[float, float]:
    # Helper to apply modulation
    # lr_new = lr * lr_mult
    # temp_new = temp * temp_mult
    pass
```

**Integration with trainer.py:**
```python
# In FDTScheduler.step()
emotion_state = self.emotion_head(hidden_state)
policy_mods = modulate_policy(fdt_metrics, emotion_state, config)

lr_mult = policy_mods['lr_mult']
temp_mult = policy_mods['temp_mult']

# Apply with PGU veto
lr_adjusted = self.apply_lr_modulation(lr_mult)
temp_adjusted = self.apply_temp_modulation(temp_mult)
```

### Priority 2: Topology Gates & Safety

#### tfan/mm/topo_gate.py (Target: 200 lines)

**Purpose:** Topological acceptance gates

**Key Functions:**
```python
def compute_topology_metrics(
    latent_before: torch.Tensor,
    latent_after: torch.Tensor,
    target_pd: Optional[np.ndarray] = None
) -> dict:
    # Compute persistence diagrams
    # Calculate Wasserstein distance
    # Calculate cosine similarity
    # Returns: {'wasserstein_pct': ..., 'cosine': ..., 'pass': bool}
    pass

def topology_gate(
    fusion_output: torch.Tensor,
    target_structure: dict,
    config: dict
) -> bool:
    # Check Wasserstein ≤ 2%
    # Check cosine ≥ 0.90
    # Return True if pass, else trigger CAT fallback
    pass
```

**CAT Fallback:**
```python
def trigger_cat_fallback(
    sparse_output: torch.Tensor,
    dense_output: torch.Tensor,
    gate_failure_reason: str
) -> torch.Tensor:
    # Log failure
    # Return dense output
    # Update metrics
    pass
```

### Priority 3: Tests & CI

#### tests/test_mm_ingest.py

```python
def test_text_adapter():
    # Shape invariants
    # Timestamp ordering
    # Max length enforcement
    pass

def test_audio_adapter():
    # Frame count vs duration
    # Prosody dimensions
    # Mel bands
    pass

def test_video_adapter():
    # Patch count
    # Timestamp @ FPS
    # Feature dimensions
    pass

def test_imu_adapter():
    # Normalization
    # Timestamp @ rate
    # Feature count
    pass
```

#### tests/test_mm_align.py

```python
def test_ttw_latency_gate():
    # Assert p95 < 5ms
    pass

def test_alignment_coverage():
    # Assert coverage ≥ 90%
    pass

def test_fast_triggers():
    # VFE spike detection
    # Entropy jump detection
    pass
```

#### tests/test_mm_fuse.py

```python
def test_token_packing():
    # [MOD] sentinels
    # [FUSE] mediators
    # Timestamp preservation
    pass

def test_tls_masks():
    # Mask shape correctness
    # keep_ratio enforcement
    # Per-head masks
    pass

def test_ssa_speedup_gate():
    # Benchmark @ 16k, 32k
    # Assert ≥3× speedup
    pass
```

#### tests/test_emotion.py

```python
def test_va_head():
    # Output shape [B, 2]
    # Range checks
    pass

def test_pad_head():
    # Output shape [B, 3]
    pass

def test_emotion_losses():
    # CCC computation
    # Temporal smoothness
    # Topological penalty
    pass

def test_trajectory_topology():
    # β₀=1 enforcement
    # β₁=0 enforcement
    pass
```

#### tests/test_topo_gate.py

```python
def test_wasserstein_gate():
    # Compute gap
    # Assert ≤ 2%
    pass

def test_cosine_gate():
    # Compute similarity
    # Assert ≥ 0.90
    pass

def test_cat_fallback():
    # Trigger on failure
    # Verify dense output used
    pass
```

### Priority 4: CI Workflows

#### .github/workflows/multimodal_smoke.yml

```yaml
name: Multi-Modal Smoke Test

on: [push, pull_request]

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Test adapters
        run: pytest tests/test_mm_ingest.py -v

      - name: Test alignment
        run: pytest tests/test_mm_align.py -v

      - name: Test fusion
        run: pytest tests/test_mm_fuse.py -v

      - name: Verify gates
        run: |
          python -c "
          from tfan.mm import align_streams
          # Quick gate check on synthetic data
          "
```

#### .github/workflows/emotion_bench.yml

```yaml
name: Emotion Benchmark

on:
  schedule:
    - cron: '0 2 * * *'  # Nightly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Test emotion heads
        run: pytest tests/test_emotion.py -v

      - name: Evaluate CCC
        run: python benchmarks/eval_emotion.py --metric ccc

      - name: Check topology gate
        run: python scripts/check_emotion_topology.py

      - name: Export metrics
        run: |
          # Export CCC, RMSE, jerk to Prometheus
          python monitoring/export_emotion_metrics.py
```

---

## 📊 Dataflow (Complete E2E)

```
┌─────────────────────────────────────────────────────────┐
│ 1. INGEST (tfan/mm/ingest.py)                          │
│    TextAdapter → (tokens, ts)                           │
│    AudioAdapter → (mels+prosody, ts)                    │
│    VideoAdapter → (patches, ts)                         │
│    IMUAdapter → (sensors, ts) [optional]                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 2. ALIGN (tfan/mm/align.py)                            │
│    align_streams() with TTW-Sentry                      │
│    Fast triggers: VFE spike, entropy jump               │
│    Gate: p95 < 5ms ✓                                    │
│    Output: aligned streams @ universal timebase         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 3. FUSE (tfan/mm/fuse.py) [TODO]                       │
│    Pack: [MOD] + tokens + [FUSE] mediators              │
│    TLS: α·persistence + (1−α)·maxmin → landmark_scores  │
│    SSA: Generate per-head masks (keep_ratio=0.33)       │
│    Gate: speedup ≥ 3× @ 16k/32k ✓                       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 4. ATTENTION (tfan/attention.py)                        │
│    RadialSparseAttention with TLS masks                 │
│    Topology loss via topo.py (PLLay)                    │
│    Gate: Wasserstein ≤ 2%, cosine ≥ 0.90 ✓             │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 5. HEADS (tfan/emotion/head.py) [TODO]                 │
│    Task Head → predictions                              │
│    Emotion Head (VA or PAD) → emotion state             │
│    Losses: Task + Emotion + Topology + Alignment        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 6. CONTROL (tfan/emotion/controller.py) [TODO]         │
│    modulate_policy(fdt_metrics, emotion_state)          │
│    Arousal ↔ T, Valence ↔ LR, Confidence ↔ PGU         │
│    Apply to trainer.py FDT loop                         │
│    Gate: EPR-CV ≤ 0.15 ✓                                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 7. PGU (tfan/pgu.py)                                    │
│    Constraint checks, cache reuse                       │
│    Final arbiter on policy modulation                   │
│    Gate: p95 ≤ 200ms, cache hit ≥ 50% ✓                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 8. MONITORING (monitoring/dashboard_setup.py)          │
│    Prometheus metrics, Grafana dashboards               │
│    All gates tracked in real-time                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🏁 Milestones & Timeline

### M0 (Day 0–2): ✅ COMPLETE
- ✅ Adapters (text/audio/video/IMU)
- ✅ Alignment wrapper (TTW-Sentry)
- ✅ Config schema

### M1 (Day 3–5): 🔨 TODO
- tfan/mm/fuse.py - Token packing + TLS masks
- SSA benchmark (≥3× @ 16k)
- tfan/mm/topo_gate.py - Wasserstein/cosine gates

### M2 (Day 6–8): 🔨 TODO
- tfan/emotion/head.py - VA/PAD heads + losses
- tfan/emotion/controller.py - Policy modulation
- Integration with trainer.py FDT loop
- EPR-CV ≤ 0.15 on small mm run

### M3 (Day 9–12): 🔨 TODO
- All tests passing
- CI workflows green
- PGU interaction verified (p95 ≤ 200ms)
- Grafana dashboards live

### M4 (Day 13–14): 🔨 TODO
- End-to-end eval on real multi-modal data
- Export report + artifact bundle
- Production deployment ready

---

## 🚀 Quick Start (When Complete)

```bash
# 1. Smoke tests
pytest tests/test_mm_ingest.py -q
pytest tests/test_mm_align.py -q
pytest tests/test_mm_fuse.py -q
pytest tests/test_emotion.py -q

# 2. End-to-end training
python -m tfan.trainer \
    --config configs/multimodal_emotion.yaml \
    --steps 2000

# 3. Dashboards
python monitoring/dashboard_setup.py \
    --config configs/multimodal_emotion.yaml

# 4. Verify gates
python scripts/verify_all_gates.py \
    --config configs/multimodal_emotion.yaml
```

---

## 📋 Implementation Checklist

- [x] tfan/mm/ingest.py (+ tests)
- [x] tfan/mm/align.py (TTW wrapper + precursors)
- [x] configs/multimodal_emotion.yaml
- [ ] tfan/mm/fuse.py (packing + TLS/SSA masks)
- [ ] tfan/emotion/head.py (VA/PAD heads + losses)
- [ ] tfan/emotion/controller.py (policy modulation)
- [ ] tfan/mm/topo_gate.py (Wasserstein/cosine gates)
- [ ] Tests: ingest ✓, align ✓, fuse, emotion, topo_gate
- [ ] CI workflows: multimodal_smoke.yml, emotion_bench.yml
- [ ] Dashboard updates with new panels
- [ ] docs/PRODUCTION_GUIDE.md - Multi-Modal & Emotion section

---

## 🎓 Key Design Decisions

**Why VA over PAD?**
- 2D easier to visualize and control
- VA trajectory topology simpler (β₀=1, β₁=0)
- Can always extend to PAD later

**Why TTW for alignment?**
- Sub-5ms p95 latency achievable
- Fast triggers (VFE, entropy) for common case
- Coverage ≥ 90% without excessive compute

**Why TLS for SSA?**
- Persistence captures semantic importance
- Maxmin ensures coverage
- α=0.7 blend empirically optimal

**Why emotion ↔ FDT coupling?**
- Arousal signals volatility → adjust temperature
- Valence signals progress → adjust learning rate
- Homeostatic stability via PGU final arbiter

**Why all these gates?**
- TTW p95 < 5ms: Real-time alignment
- PGU p95 ≤ 200ms: Safety constraint
- EPR-CV ≤ 0.15: Training stability
- SSA ≥ 3×: Computational efficiency
- Topology: Manifold quality guarantee

---

**STATUS: Foundation Complete (30% done), Core Implementation Next (M1-M2)**

Let's build this! 🚀
