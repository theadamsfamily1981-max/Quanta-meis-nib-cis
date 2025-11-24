"""Training-ready JAX implementation of the Topological Field-Adaptive Network."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

Array = jax.Array


@dataclass
class TFANConfig:
    input_dim: int = 256
    topo_dim: int = 48
    hidden_dim: int = 512
    latent_dim: int = 256
    output_dim: int = 12
    num_blocks: int = 4
    dropout_rate: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2


class PersistenceEncoder(nn.Module):
    features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, persistence: Array, training: bool) -> Array:
        x = nn.Dense(self.features * 2)(persistence)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.features)(x)
        return nn.sigmoid(x)


class FieldKernel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: Array, laplacian: Array) -> Array:
        projected = nn.Dense(self.features)(x)
        laplacian_summary = jnp.mean(laplacian, axis=-1)
        laplacian_emb = nn.Dense(self.features)(laplacian_summary)
        return nn.LayerNorm()(projected + laplacian_emb)


class TopologicalBlock(nn.Module):
    hidden_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        x: Array,
        topo_gate: Array,
        laplacian: Array,
        training: bool,
    ) -> Array:
        kernel = FieldKernel(self.hidden_dim)(x, laplacian)
        gated = kernel * topo_gate
        gated = nn.Dense(self.hidden_dim)(gated)
        gated = nn.gelu(gated)
        gated = nn.Dropout(self.dropout_rate)(gated, deterministic=not training)
        return nn.LayerNorm()(x + gated)


class ReadoutHead(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x: Array, training: bool) -> Array:
        x = nn.Dense(x.shape[-1] // 2)(x)
        x = nn.gelu(x)
        x = nn.Dropout(0.1)(x, deterministic=not training)
        x = nn.Dense(self.output_dim)(x)
        return x


class TopologicalFieldAdaptiveNetwork(nn.Module):
    config: TFANConfig

    def setup(self) -> None:
        self.encoder = nn.Dense(self.config.hidden_dim)
        self.persistence_encoder = PersistenceEncoder(
            features=self.config.hidden_dim, dropout_rate=self.config.dropout_rate
        )
        self.blocks = [
            TopologicalBlock(self.config.hidden_dim, self.config.dropout_rate)
            for _ in range(self.config.num_blocks)
        ]
        self.readout = ReadoutHead(self.config.output_dim)

    def __call__(
        self,
        inputs: Array,
        persistence: Array,
        laplacian: Array,
        training: bool,
    ) -> Array:
        x = self.encoder(inputs)
        x = nn.gelu(x)
        topo_gate = self.persistence_encoder(persistence, training)
        for block in self.blocks:
            x = block(x, topo_gate, laplacian, training)
        logits = self.readout(x, training)
        return logits


def create_train_state(
    key: Array, config: TFANConfig
) -> Tuple[train_state.TrainState, Array]:
    model = TopologicalFieldAdaptiveNetwork(config)

    @jax.jit
    def init_model(rng: Array) -> Dict[str, Any]:
        sample_x = jnp.zeros((1, config.input_dim))
        sample_p = jnp.zeros((1, config.topo_dim))
        sample_l = jnp.eye(config.input_dim).reshape(1, config.input_dim, config.input_dim)
        variables = model.init(rng, sample_x, sample_p, sample_l, training=False)
        return variables

    params_key, dropout_key = jax.random.split(key)
    variables = init_model(params_key)
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    return state, dropout_key


def compute_laplacian(batch: Array) -> Array:
    cov = batch[:, :, None] * batch[:, None, :]
    norm = jnp.linalg.norm(cov, axis=(-2, -1), keepdims=True) + 1e-6
    adjacency = cov / norm
    degree = jnp.sum(adjacency, axis=-1)
    identity = jnp.eye(adjacency.shape[-1])
    degree_matrix = jnp.einsum("bi,ij->bij", degree, identity)
    return degree_matrix - adjacency


def loss_fn(
    params: Dict[str, Any],
    state: train_state.TrainState,
    batch: Array,
    persistence: Array,
    laplacian: Array,
    labels: Array,
    rng: Array,
) -> Tuple[Array, Tuple[Array, Array]]:
    logits = state.apply_fn(
        {"params": params}, batch, persistence, laplacian, training=True, rngs={"dropout": rng}
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, (loss, acc)


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: Array,
    persistence: Array,
    laplacian: Array,
    labels: Array,
    rng: Array,
) -> Tuple[train_state.TrainState, Dict[str, Array], Array]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (scalar_loss, acc)), grads = grad_fn(
        state.params, state, batch, persistence, laplacian, labels, rng
    )
    state = state.apply_gradients(grads=grads)
    _, next_rng = jax.random.split(rng)
    return state, {"loss": scalar_loss, "accuracy": acc}, next_rng


def generate_synthetic_data(
    key: Array, config: TFANConfig, batch_size: int
) -> Tuple[Array, Array, Array, Array]:
    key_x, key_p, key_noise, key_labels = jax.random.split(key, 4)
    features = jax.random.normal(key_x, (batch_size, config.input_dim))
    persistence = jax.random.uniform(key_p, (batch_size, config.topo_dim))
    laplacian = compute_laplacian(features)
    logits = features[:, : config.output_dim]
    logits += 0.1 * jax.random.normal(key_noise, logits.shape)
    labels = jnp.argmax(logits, axis=-1)
    labels = (labels + jax.random.randint(key_labels, labels.shape, 0, 2)) % config.output_dim
    return features, persistence, laplacian, labels


def evaluate(
    state: train_state.TrainState,
    config: TFANConfig,
    rng: Array,
    batch_size: int,
) -> Dict[str, float]:
    features, persistence, laplacian, labels = generate_synthetic_data(rng, config, batch_size)
    logits = state.apply_fn(
        {"params": state.params}, features, persistence, laplacian, training=False
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def count_parameters(params: Dict[str, Any]) -> int:
    return int(sum(jnp.prod(jnp.array(v.shape)) for v in jax.tree_util.tree_leaves(params)))


def run_training(args: argparse.Namespace) -> None:
    config = TFANConfig(
        input_dim=args.input_dim,
        topo_dim=args.topo_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    rng = jax.random.PRNGKey(args.seed)
    state, dropout_key = create_train_state(rng, config)

    print(f"Initial parameter count: {count_parameters(state.params)}")

    for epoch in range(args.epochs):
        rng, batch_key = jax.random.split(rng)
        features, persistence, laplacian, labels = generate_synthetic_data(
            batch_key, config, args.batch_size
        )
        state, metrics, dropout_key = train_step(
            state, features, persistence, laplacian, labels, dropout_key
        )
        print(
            f"Epoch {epoch + 1:02d} | loss={float(metrics['loss']):.4f} "
            f"accuracy={float(metrics['accuracy']):.4f}"
        )

    eval_metrics = evaluate(state, config, rng, args.batch_size)
    print("Validation metrics:", eval_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=12)
    parser.add_argument("--topo-dim", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run_training(parser.parse_args())
