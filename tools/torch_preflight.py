#!/usr/bin/env python3
# Torch Preflight: verifies CUDA, cuDNN, GPUs, does a tiny compute & train step,
# and writes torch_preflight_results.json alongside this script.
import json, sys, time, subprocess, platform

def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"

def _smi_query():
    q = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"]
    return _run(q)

def main():
    info = {
        "python": sys.version.replace("\n", " ") ,
        "platform": platform.platform(),
        "nvidia_smi": _smi_query(),
        "notes": "If cuda_available=False or cudnn_available=False, install a GPU wheel that matches your CUDA runtime and driver."
    }
    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "compiled_with_cuda": bool(torch.backends.cuda.is_built()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        }
        # Enumerate GPUs
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_GB": round(props.total_memory / (1024**3), 2),
                    "sm_major": props.major,
                    "sm_minor": props.minor,
                    "multiprocessors": props.multi_processor_count
                })
        info["gpus"] = gpus

        # Quick compute test
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        x = torch.randn(4096, 4096, device=device)
        y = torch.randn(4096, 4096, device=device)
        z = x @ y
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mm_ms = (time.time() - t0) * 1000.0

        # Tiny train step
        model = torch.nn.Linear(4096, 1024).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        t1 = time.time()
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            pred = model(z)
            loss = loss_fn(pred, torch.randn_like(pred))
            loss.backward()
            opt.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_ms = (time.time() - t1) * 1000.0

        info["sanity"] = {
            "device_used": device,
            "matmul_4k_ms": round(mm_ms, 2),
            "tiny_train_5_steps_ms": round(train_ms, 2),
            "loss_last": float(loss.item())
        }

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"

    out_path = "torch_preflight_results.json"
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
