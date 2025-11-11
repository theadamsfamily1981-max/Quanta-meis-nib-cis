import math
import random
from typing import Sequence, Callable

def _to_list(x: Sequence[float]) -> list:
  return list(x)

def _norm(vec: Sequence[float]) -> float:
  return math.sqrt(sum(v * v for v in vec))

def finite_diff_grad(f: Callable[[Sequence[float]], float], x: Sequence[float], eps: float = 1e-4):
  x_list = _to_list(x)
  grad = []
  for i in range(len(x_list)):
    x1 = x_list.copy(); x1[i] += eps
    x2 = x_list.copy(); x2[i] -= eps
    grad.append((f(x1) - f(x2)) / (2 * eps))
  return grad

def monotonicity_check(f_true: Callable[[Sequence[float]], float],
                       f_proxy: Callable[[Sequence[float]], float],
                       x: Sequence[float], delta: float = 1e-3, trials: int = 32):
  x_list = _to_list(x)
  ok = 0
  for _ in range(trials):
    direction = [random.gauss(0.0, 1.0) for _ in x_list]
    norm = _norm(direction) or 1.0
    direction = [d / norm for d in direction]
    x2 = [xi - delta * di for xi, di in zip(x_list, direction)]
    if (f_true(x2) <= f_true(x_list)) == (f_proxy(x2) <= f_proxy(x_list)):
      ok += 1
  return ok / float(trials)

def gradient_nrmse(grad_true: Sequence[float], grad_proxy: Sequence[float]):
  diff_sq = sum((gt - gp) ** 2 for gt, gp in zip(grad_true, grad_proxy))
  denom = math.sqrt(sum(gt ** 2 for gt in grad_true)) + 1e-12
  return math.sqrt(diff_sq) / denom

def validate_proxy(f_true: Callable[[Sequence[float]], float],
                   f_proxy: Callable[[Sequence[float]], float],
                   x0: Sequence[float], analytic_grad_proxy: Callable[[Sequence[float]], Sequence[float]],
                   eps_nrmse: float = 0.03, mono_frac: float = 0.8):
  g_fd = finite_diff_grad(f_proxy, x0)
  g_an = _to_list(analytic_grad_proxy(x0))
  nrmse = gradient_nrmse(g_fd, g_an)
  mono = monotonicity_check(f_true, f_proxy, x0)
  return {
    "nrmse": float(nrmse),
    "monotonicity_agree_frac": float(mono),
    "pass": (nrmse <= eps_nrmse) and (mono >= mono_frac)
  }
