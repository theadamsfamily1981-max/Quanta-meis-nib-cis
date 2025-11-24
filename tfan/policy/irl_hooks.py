"""
Lightweight IRL hooks (stubs):
 - log_trajectory: push (state, action, outcome) tuples
 - fit_reward_model: simple linear reward on features
 - infer_preferences: map recent outcomes to preference priors
"""
import numpy as np


class IRLBuffer:
  def __init__(self, maxlen=5000):
    self.S, self.A, self.R = [], [], []
    self.maxlen = maxlen

  def log_trajectory(self, states, actions, rewards):
    for s, a, r in zip(states, actions, rewards):
      self.S.append(np.asarray(s))
      self.A.append(np.asarray(a))
      self.R.append(float(r))
    if len(self.S) > self.maxlen:
      cut = len(self.S) - self.maxlen
      self.S = self.S[cut:]
      self.A = self.A[cut:]
      self.R = self.R[cut:]


class LinearRewardIRL:
  def __init__(self):
    self.w = None

  def fit_reward_model(self, feats, returns, l2=1e-3):
    # ridge regression: w = (X^T X + l2 I)^-1 X^T y
    X = np.asarray(feats, dtype=float)
    y = np.asarray(returns, dtype=float)
    XtX = X.T @ X + l2 * np.eye(X.shape[1])
    self.w = np.linalg.solve(XtX, X.T @ y)
    return self.w

  def infer_preferences(self, feat_importances, temperature=1.0):
    # Map normalized positive importances -> preference vector
    v = np.maximum(feat_importances, 0.0)
    if v.sum() == 0:
      v += 1.0
    v = v / v.sum()
    # temperature smoothing
    v = v ** (1.0 / max(1e-6, temperature))
    v = v / v.sum()
    return v
