# QUANTA / MEIS / NIB — Google Drive + Runbook (for You & Claude)

This document gives you and Claude a single checklist to:
1) Organize data on Google Drive
2) Connect/share access securely
3) Set environment variables
4) Run local + CI benches (including the new INNOV priority suite)
5) Save artifacts back to Drive

> If you connect Drive to ChatGPT later, I can read from it directly; until then, follow these steps.

---

## A) Google Drive: Folder Structure & Sharing
````
QUANTA_Projects/
  ├─ datasets/
  │   ├─ MELD/
  │   ├─ IEMOCAP/
  │   └─ MOSEI/                 (optional)
  ├─ runs/                      (bench outputs, logs, JSONs, SVGs)
  ├─ docs/                      (reports, dashboards, PDFs)
  └─ keys/                      (API keys, private configs — DO NOT share)
````
1. Right‑click **QUANTA_Projects/** → **Share**.
   - Add your secondary accounts (and Claude’s workspace address if applicable).
   - Permissions: *Editor* for you; *Viewer/Commenter* for assistants.
   - Keep **Anyone with the link** = *Off*.
2. Create a dated run folder, e.g. `QUANTA_Projects/runs/2025-11-02/`.
3. Put private credentials **only** in `keys/` (never commit to GitHub).

## B) Map Local Paths ↔ Drive
If using Drive for Desktop (Win/macOS) or rclone (Linux), ensure:
- `MELD_ROOT → /path/to/Drive/QUANTA_Projects/datasets/MELD`
- `IEMOCAP_ROOT → /path/to/Drive/QUANTA_Projects/datasets/IEMOCAP`

Symlinks (Linux/macOS):
```bash
ln -s "/path/to/Drive/QUANTA_Projects/datasets/MELD"    "$HOME/data/MELD"
ln -s "/path/to/Drive/QUANTA_Projects/datasets/IEMOCAP" "$HOME/data/IEMOCAP"
```

## C) Environment Variables
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export MELD_ROOT="$HOME/data/MELD"
export IEMOCAP_ROOT="$HOME/data/IEMOCAP"
export CUDA_VISIBLE_DEVICES="0,1"       # optional; set after gpu_selector.py
export QMNC_BACKEND="torch"             # fallback to numpy if torch not installed
```
Reload: `source ~/.bashrc` (or `~/.zshrc`).

## D) Clone & Prep the Repo
```bash
git clone https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis.git
cd Quanta-meis-nib-cis
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
# Optional: GPU check
python tools/torch_preflight.py
python tools/gpu_selector.py
```

## E) Quick Local Smoke
```bash
pytest -q              # run tests
make catalog           # 45-experiment manifest
python scripts/run_innov_priority.py  # INNOV (001,016,036) — CPU-safe
make rc                # production suite demo (best-effort)
```
Artifacts appear in `./reports/` or as printed JSON.

## F) Save Artifacts Back to Drive
```bash
mkdir -p "/path/to/Drive/QUANTA_Projects/runs/2025-11-02"
cp -r reports "/path/to/Drive/QUANTA_Projects/runs/2025-11-02/reports"
# optional:
cp -r docs    "/path/to/Drive/QUANTA_Projects/runs/2025-11-02/docs"
```

## G) GitHub Bench (no GPU/data, label-triggered)
- In a PR, add label: **bench**
- Or Actions → **Bench (INNOV priority)** → Run workflow
- Artifact: **innov-priority-json**

## H) Claude: Suggested Prompts
**Environment & CUDA**
```text
Claude, run these and paste outputs:
python tools/torch_preflight.py
python tools/gpu_selector.py
cat torch_preflight_results.json
```
**INNOV Priority Bench (local)**
```text
Claude, in repo root, run:
pytest -q
python scripts/run_innov_priority.py
Show me the JSON and save to ./reports/innov_priority.json
```
**Save to Drive**
```text
Claude, copy ./reports to my Drive at:
/path/to/Drive/QUANTA_Projects/runs/2025-11-02/reports
Then list the copied files.
```
**Production Suite (datasets present)**
```text
Claude, ensure MELD_ROOT and IEMOCAP_ROOT are set, then run:
source ~/.bashrc  # if needed
make rc
Zip ./reports and upload to QUANTA_Projects/runs/2025-11-02/
```

## I) Drive Connection Notes for ChatGPT
- I can access your Drive **read-only** right now; I’ll need write permission via this connector to create/edit files.
- Until then, I can generate files here for download; you can upload them to Drive.

## J) Troubleshooting
- **Torch CUDA not available**: install a CUDA‑matched PyTorch; re‑run `tools/torch_preflight.py`.
- **Missing datasets**: set `MELD_ROOT/IEMOCAP_ROOT` correctly; verify paths.
- **CI failing**: `ruff check .`, `black --check .`, `pytest -q`; validate JSON with `jq . reports/innov_priority.json`.
- **Bench artifact missing**: ensure label **bench** or manual run; check Actions logs.

## K) Minimal Checklist
- [ ] Create Drive folder structure
- [ ] Share with correct accounts (private)
- [ ] Map `MELD_ROOT/IEMOCAP_ROOT`
- [ ] Clone repo + install dev deps
- [ ] Run preflight + gpu_selector
- [ ] Run tests + `scripts/run_innov_priority.py`
- [ ] Copy `./reports` to Drive `runs/YYYY-MM-DD`
- [ ] (Optional) Label PR with **bench**
