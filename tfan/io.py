import json, time, uuid, yaml
from pathlib import Path
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)

def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def new_artifact(name: str, suffix=".json") -> Path:
    ensure_dir("artifacts")
    return Path("artifacts") / f"{name}_{ts()}_{uuid.uuid4().hex[:6]}{suffix}"

def write_json(path: str | Path, obj: Dict):
    Path(path).write_text(json.dumps(obj, indent=2))
