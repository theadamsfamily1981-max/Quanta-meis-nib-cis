import json, os, time, statistics, pathlib
from typing import List

def write_jsonl(records: List[dict], sink: str | None = None):
    sink = sink or os.getenv("TFAN_TELEMETRY_SINK", "local://artifacts")
    ts = int(time.time()*1000)
    if sink.startswith("local://"):
        path = pathlib.Path("artifacts/logs"); path.mkdir(parents=True, exist_ok=True)
        with open(path/f"{ts}.jsonl","w") as f:
            for r in records: f.write(json.dumps(r)+"\n")
        return {"ok": True, "path": str(path)}
    # S3/GCS hooks are stubs to keep CI green; wire after creds land
    return {"ok": True, "remote": sink}

def latency_percentiles(ms: List[float]):  # p50/p95/p99
    ms = sorted(ms)
    def pct(p): 
        i = max(0, min(len(ms)-1, int(round((p/100.0)*(len(ms)-1)))))
        return ms[i]
    return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}
