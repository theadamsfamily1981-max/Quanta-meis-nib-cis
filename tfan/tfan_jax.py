"""JAX implementation of the Temporal Fractal Attention Network (T-FAN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


@dataclass
class TFANConfig:
    hidden_size: int = 64
    fractal_depth: int = 3


@dataclass
class TFANParameters:
    query_kernel: jnp.ndarray
    key_kernel: jnp.ndarray
    value_kernel: jnp.ndarray


@dataclass
class TFANState:
    params: TFANParameters


def initialize_tfan(rng: jax.random.KeyArray, input_dim: int, config: TFANConfig) -> TFANState:
    """Initialise T-FAN parameters with Xavier uniform weights."""

    limit = jnp.sqrt(6.0 / (input_dim + config.hidden_size))
    q_kernel = jax.random.uniform(rng, (input_dim, config.hidden_size), minval=-limit, maxval=limit)
    k_kernel = jax.random.uniform(rng, (input_dim, config.hidden_size), minval=-limit, maxval=limit)
    v_kernel = jax.random.uniform(rng, (input_dim, config.hidden_size), minval=-limit, maxval=limit)
    return TFANState(TFANParameters(q_kernel, k_kernel, v_kernel))


def _attention(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    scores = jnp.matmul(query, key.T) / jnp.sqrt(query.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, value)


def build_tfan_apply(config: TFANConfig) -> Callable[[TFANState, jnp.ndarray], Tuple[jnp.ndarray, TFANState]]:
    """Create an apply function compatible with simple training loops."""

    def apply(state: TFANState, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, TFANState]:
        x = inputs
        params = state.params
        residual = x
        for _ in range(config.fractal_depth):
            query = jnp.matmul(x, params.query_kernel)
            key = jnp.matmul(x, params.key_kernel)
            value = jnp.matmul(x, params.value_kernel)
            x = residual + _attention(query, key, value)
        return x, state

    return jax.jit(apply)
