"""Prototype implementation of the Topological Field-Adaptive Network (T-FAN).

This module intentionally keeps the API surface minimal so researchers can read
and reason about the mechanics before transitioning to the full training-ready
reference implementation in :mod:`tfan_jax_working`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class TFANConfig:
    """Configuration for the prototype network."""

    input_dim: int = 128
    hidden_dim: int = 256
    topo_dim: int = 32
    output_dim: int = 10
    dropout_rate: float = 0.0


def _glorot_uniform(key: jax.Array, shape: Tuple[int, int]) -> Array:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _dense(params: Dict[str, Array], inputs: Array) -> Array:
    return inputs @ params["kernel"] + params["bias"]


def _layer_norm(x: Array, eps: float = 1e-6) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _topological_gate(params: Dict[str, Array], persistence: Array) -> Array:
    logits = persistence @ params["kernel"] + params["bias"]
    return jax.nn.sigmoid(logits)


def init_parameters(key: jax.Array, config: TFANConfig) -> Dict[str, Dict[str, Array]]:
    """Initialise the prototype network parameters."""

    key1, key2, key3, key4 = jax.random.split(key, 4)
    params = {
        "encoder": {
            "kernel": _glorot_uniform(key1, (config.input_dim, config.hidden_dim)),
            "bias": jnp.zeros((config.hidden_dim,)),
        },
        "adapter": {
            "kernel": _glorot_uniform(key2, (config.hidden_dim, config.hidden_dim)),
            "bias": jnp.zeros((config.hidden_dim,)),
        },
        "gate": {
            "kernel": _glorot_uniform(key3, (config.topo_dim, config.hidden_dim)),
            "bias": jnp.zeros((config.hidden_dim,)),
        },
        "classifier": {
            "kernel": _glorot_uniform(key4, (config.hidden_dim, config.output_dim)),
            "bias": jnp.zeros((config.output_dim,)),
        },
    }
    return params


def forward(
    params: Dict[str, Dict[str, Array]],
    inputs: Array,
    persistence: Array,
) -> Array:
    """Run a forward pass through the prototype network."""

    hidden = _dense(params["encoder"], inputs)
    hidden = jax.nn.gelu(hidden)
    hidden = _layer_norm(hidden)

    topo_gate = _topological_gate(params["gate"], persistence)
    adapted = hidden * topo_gate
    adapted = _dense(params["adapter"], adapted)
    adapted = jax.nn.gelu(adapted)

    logits = _dense(params["classifier"], adapted)
    return logits


def cross_entropy_loss(logits: Array, labels: Array) -> Array:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1).mean()


def accuracy(logits: Array, labels: Array) -> Array:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


def count_parameters(params: Dict[str, Dict[str, Array]]) -> int:
    return int(sum(v.size for layer in params.values() for v in layer.values()))


def main() -> None:
    """Runs a toy forward pass for smoke testing the prototype."""

    config = TFANConfig()
    key = jax.random.PRNGKey(0)
    params = init_parameters(key, config)

    batch = jax.random.normal(key, (8, config.input_dim))
    persistence = jax.random.uniform(key, (8, config.topo_dim))
    labels = jax.random.randint(key, (8,), 0, config.output_dim)

    logits = forward(params, batch, persistence)
    loss = cross_entropy_loss(logits, labels)
    acc = accuracy(logits, labels)

    print(f"Prototype loss: {loss:.3f}")
    print(f"Prototype accuracy: {acc:.3f}")
    print(f"Parameter count: {count_parameters(params)}")


if __name__ == "__main__":
    main()
