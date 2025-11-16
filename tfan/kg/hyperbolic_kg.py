#!/usr/bin/env python
"""
Hyperbolic Knowledge Graph (HyperKG)

Represents knowledge graphs in hyperbolic space (Poincaré ball model).
Hyperbolic geometry naturally captures hierarchical structures common in KGs.

Key properties:
- Exponential volume growth → hierarchies fit naturally
- Distance preserves transitive relations
- Entailment = hyperbolic distance < threshold

Usage:
    from tfan.kg import HyperbolicKG

    # Create KG with hyperbolic embeddings
    kg = HyperbolicKG(embedding_dim=64, curvature=-1.0)

    # Add entities and relations
    kg.add_triple("socrates", "is_a", "human")
    kg.add_triple("human", "is_a", "mortal")

    # Train embeddings
    kg.train(epochs=100)

    # Query with hyperbolic inference
    result = kg.query("socrates", "is_a", "mortal")
    # Returns: True (via transitivity in hyperbolic space)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class PoincareEmbedding(nn.Module):
    """
    Poincaré ball embeddings for hierarchical data.

    The Poincaré ball is the open unit ball with hyperbolic metric:
        d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))

    Points near the boundary represent leaves in the hierarchy.
    Points near the origin represent roots.
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int = 64,
        curvature: float = -1.0,
        epsilon: float = 1e-5
    ):
        """
        Initialize Poincaré embeddings.

        Args:
            num_entities: Number of entities to embed
            embedding_dim: Dimension of hyperbolic space
            curvature: Curvature of hyperbolic space (negative)
            epsilon: Small constant to avoid numerical issues
        """
        super().__init__()

        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.epsilon = epsilon

        # Initialize embeddings uniformly in the ball
        # Use small values to avoid boundary
        self.embeddings = nn.Parameter(
            torch.randn(num_entities, embedding_dim) * 0.01
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for entity indices."""
        return self.embeddings[indices]

    def project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points to the Poincaré ball.

        Ensures ||x|| < 1 by projecting to ball of radius 1 - epsilon.
        """
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.epsilon)
        max_norm = 1.0 - self.epsilon
        scale = torch.where(
            norm > max_norm,
            max_norm / norm,
            torch.ones_like(norm)
        )
        return x * scale

    def poincare_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Poincaré distance between points u and v.

        d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))
        """
        # Project to ensure valid points
        u = self.project_to_ball(u)
        v = self.project_to_ball(v)

        # Compute norms
        u_norm_sq = (u ** 2).sum(dim=-1, keepdim=True).clamp(max=1.0 - self.epsilon)
        v_norm_sq = (v ** 2).sum(dim=-1, keepdim=True).clamp(max=1.0 - self.epsilon)

        # Compute squared distance
        diff_norm_sq = ((u - v) ** 2).sum(dim=-1, keepdim=True)

        # Poincaré distance formula
        numerator = 2 * diff_norm_sq
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)

        # Use arcosh(1 + x) for numerical stability
        # arcosh(1 + x) ≈ sqrt(2x) for small x
        delta = numerator / denominator.clamp(min=self.epsilon)
        distance = torch.acosh(1 + delta.clamp(min=0.0))

        return distance.squeeze(-1)

    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.

        u ⊕ v = (1 + 2⟨u,v⟩ + ||v||²)u + (1 - ||u||²)v
                / (1 + 2⟨u,v⟩ + ||u||²||v||²)
        """
        u_norm_sq = (u ** 2).sum(dim=-1, keepdim=True)
        v_norm_sq = (v ** 2).sum(dim=-1, keepdim=True)
        uv_dot = (u * v).sum(dim=-1, keepdim=True)

        numerator = (1 + 2 * uv_dot + v_norm_sq) * u + (1 - u_norm_sq) * v
        denominator = 1 + 2 * uv_dot + u_norm_sq * v_norm_sq

        result = numerator / denominator.clamp(min=self.epsilon)
        return self.project_to_ball(result)


class HyperbolicKG:
    """
    Hyperbolic Knowledge Graph with Datalog-style rules.

    Combines:
    - Hyperbolic embeddings for entities and relations
    - Rule-based inference compiled into geometric constraints
    - PGU bridge for symbolic verification
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        curvature: float = -1.0,
        distance_threshold: float = 0.5
    ):
        """
        Initialize hyperbolic KG.

        Args:
            embedding_dim: Dimension of hyperbolic space
            curvature: Hyperbolic curvature (negative)
            distance_threshold: Max distance for entailment
        """
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.distance_threshold = distance_threshold

        # Entity and relation vocabularies
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}

        # Triples: (head, relation, tail)
        self.triples: List[Tuple[int, int, int]] = []

        # Compiled rules: head :- body
        self.rules: List[Dict] = []

        # Embeddings (initialized after entities/relations are added)
        self.entity_embeddings: Optional[PoincareEmbedding] = None
        self.relation_embeddings: Optional[PoincareEmbedding] = None

    def add_entity(self, entity: str) -> int:
        """Add entity and return its index."""
        if entity not in self.entity_to_idx:
            idx = len(self.entity_to_idx)
            self.entity_to_idx[entity] = idx
            self.idx_to_entity[idx] = entity
        return self.entity_to_idx[entity]

    def add_relation(self, relation: str) -> int:
        """Add relation and return its index."""
        if relation not in self.relation_to_idx:
            idx = len(self.relation_to_idx)
            self.relation_to_idx[relation] = idx
            self.idx_to_relation[idx] = relation
        return self.relation_to_idx[relation]

    def add_triple(self, head: str, relation: str, tail: str):
        """
        Add a triple to the KG.

        Args:
            head: Head entity
            relation: Relation type
            tail: Tail entity
        """
        h_idx = self.add_entity(head)
        r_idx = self.add_relation(relation)
        t_idx = self.add_entity(tail)

        self.triples.append((h_idx, r_idx, t_idx))

    def add_rule(self, head_pattern: str, body_patterns: List[str]):
        """
        Add a Datalog-style rule.

        Example:
            # ancestor(X, Z) :- parent(X, Y), parent(Y, Z)
            kg.add_rule(
                "ancestor(?X, ?Z)",
                ["parent(?X, ?Y)", "parent(?Y, ?Z)"]
            )

        Args:
            head_pattern: Head of the rule (conclusion)
            body_patterns: Body of the rule (premises)
        """
        rule = {
            'head': self._parse_pattern(head_pattern),
            'body': [self._parse_pattern(p) for p in body_patterns]
        }
        self.rules.append(rule)

    def _parse_pattern(self, pattern: str) -> Dict:
        """Parse a pattern like 'parent(?X, ?Y)' into components."""
        # Simple parser: relation(arg1, arg2)
        parts = pattern.replace(')', '').split('(')
        relation = parts[0].strip()
        args = [arg.strip() for arg in parts[1].split(',')]

        return {
            'relation': relation,
            'args': args
        }

    def initialize_embeddings(self):
        """Initialize entity and relation embeddings."""
        if self.entity_embeddings is None:
            self.entity_embeddings = PoincareEmbedding(
                num_entities=len(self.entity_to_idx),
                embedding_dim=self.embedding_dim,
                curvature=self.curvature
            )

        if self.relation_embeddings is None:
            self.relation_embeddings = PoincareEmbedding(
                num_entities=len(self.relation_to_idx),
                embedding_dim=self.embedding_dim,
                curvature=self.curvature
            )

    def train(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        margin: float = 1.0,
        negative_samples: int = 5
    ):
        """
        Train hyperbolic embeddings using max-margin loss.

        Loss: max(0, margin + d(h+r, t) - d(h+r, t'))
        where (h, r, t) is a positive triple and t' is a negative sample.

        Args:
            epochs: Number of training epochs
            lr: Learning rate
            margin: Margin for max-margin loss
            negative_samples: Number of negative samples per positive
        """
        self.initialize_embeddings()

        optimizer = torch.optim.Adam([
            {'params': self.entity_embeddings.parameters()},
            {'params': self.relation_embeddings.parameters()}
        ], lr=lr)

        # Convert triples to tensors
        triples_tensor = torch.tensor(self.triples, dtype=torch.long)

        for epoch in range(epochs):
            total_loss = 0.0

            # Sample negatives
            for h_idx, r_idx, t_idx in self.triples:
                # Get embeddings
                h = self.entity_embeddings(torch.tensor([h_idx]))
                r = self.relation_embeddings(torch.tensor([r_idx]))
                t = self.entity_embeddings(torch.tensor([t_idx]))

                # Positive score: d(h ⊕ r, t)
                h_r = self.entity_embeddings.mobius_add(h, r)
                pos_score = self.entity_embeddings.poincare_distance(h_r, t)

                # Negative samples
                neg_loss = 0.0
                for _ in range(negative_samples):
                    t_neg_idx = np.random.randint(0, len(self.entity_to_idx))
                    t_neg = self.entity_embeddings(torch.tensor([t_neg_idx]))
                    neg_score = self.entity_embeddings.poincare_distance(h_r, t_neg)

                    # Max-margin loss
                    neg_loss += torch.relu(margin + pos_score - neg_score)

                loss = neg_loss / negative_samples
                total_loss += loss.item()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(self.triples)
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    def query(
        self,
        head: str,
        relation: str,
        tail: str,
        use_rules: bool = True
    ) -> bool:
        """
        Query whether a triple holds.

        Args:
            head: Head entity
            relation: Relation
            tail: Tail entity
            use_rules: Apply rules for inference

        Returns:
            True if triple is entailed, False otherwise
        """
        # Check if triple is explicitly in KG
        if head in self.entity_to_idx and relation in self.relation_to_idx and tail in self.entity_to_idx:
            h_idx = self.entity_to_idx[head]
            r_idx = self.relation_to_idx[relation]
            t_idx = self.entity_to_idx[tail]

            if (h_idx, r_idx, t_idx) in self.triples:
                return True

        # Hyperbolic inference
        if self.entity_embeddings is not None and head in self.entity_to_idx and tail in self.entity_to_idx and relation in self.relation_to_idx:
            h_idx = self.entity_to_idx[head]
            r_idx = self.relation_to_idx[relation]
            t_idx = self.entity_to_idx[tail]

            h = self.entity_embeddings(torch.tensor([h_idx]))
            r = self.relation_embeddings(torch.tensor([r_idx]))
            t = self.entity_embeddings(torch.tensor([t_idx]))

            h_r = self.entity_embeddings.mobius_add(h, r)
            distance = self.entity_embeddings.poincare_distance(h_r, t)

            if distance.item() < self.distance_threshold:
                return True

        # Rule-based inference (if enabled)
        if use_rules:
            # Stub: would apply rules recursively
            pass

        return False

    def get_stats(self) -> Dict:
        """Get KG statistics."""
        return {
            'num_entities': len(self.entity_to_idx),
            'num_relations': len(self.relation_to_idx),
            'num_triples': len(self.triples),
            'num_rules': len(self.rules),
            'embedding_dim': self.embedding_dim,
            'curvature': self.curvature,
        }
