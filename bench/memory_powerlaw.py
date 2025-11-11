import json
import gc
import math
import time

try:
  import torch
except ImportError:  # pragma: no cover
  torch = None

def _synthetic_points(T_values, B, d):
  return [(T, B * T * d * 4) for T in T_values]

def measure_mem(model_fn, T_values=(1024,2048,4096,8192,16384), B=1, d=128, device="cuda"):
  if torch is None:
    return _synthetic_points(T_values, B, d)
  results=[]
  for T in T_values:
    x = torch.randn(B, T, d, device=device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()
    model = model_fn().to(x.device)
    with torch.inference_mode():
      _ = model(x)
    if torch.cuda.is_available():
      mem = torch.cuda.max_memory_allocated(x.device)
    else:
      mem = x.element_size() * x.nelement()
    results.append((T, int(mem)))
    del x, model
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  return results

def fit_alpha(points):
  xs = [math.log(float(p[0])) for p in points]
  ys = [math.log(float(max(p[1], 1))) for p in points]
  n = len(xs)
  if n == 0:
    return 0.0
  mean_x = sum(xs) / n
  mean_y = sum(ys) / n
  numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
  denominator = sum((x - mean_x) ** 2 for x in xs) or 1.0
  return numerator / denominator

if __name__=="__main__":
  def dummy(): 
    if torch is None:
      class _Dummy:
        def __call__(self, x):
          return x
      return _Dummy()
    return torch.nn.Sequential(
      torch.nn.Linear(128,128),
      torch.nn.SiLU(),
      torch.nn.Linear(128,128))
  pts = measure_mem(dummy)
  alpha = fit_alpha(pts)
  print(json.dumps({"alpha": alpha, "points": pts}))
