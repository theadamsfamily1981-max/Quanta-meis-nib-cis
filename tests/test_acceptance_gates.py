import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tfan.epr.metrics import coefficient_of_variation
from tfan.validators.objective_validator import gradient_nrmse

def test_epr_cv_gate():
  cv = coefficient_of_variation([1, 1.1, 0.9, 1.05, 0.95])
  assert cv <= 0.15

def test_grad_nrmse_gate():
  g_true = [1.0, 0.0, -1.0]
  g_proxy = [0.99, 0.01, -0.98]
  assert gradient_nrmse(g_true, g_proxy) <= 0.03
