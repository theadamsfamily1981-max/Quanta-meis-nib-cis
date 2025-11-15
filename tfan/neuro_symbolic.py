"""
Neural-Symbolic Integration for TFAN.
Combines neural networks with symbolic reasoning for enhanced interpretability.

Integrates:
- Logic programming layers
- Symbolic rule learning
- Differentiable logic
- Knowledge base integration
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import re

from .pgu import PGUCache  # Reuse Z3 solver integration


@dataclass
class LogicRule:
    """Symbolic logic rule."""
    premise: str  # e.g., "X > 0 AND Y > 0"
    conclusion: str  # e.g., "Z > 0"
    confidence: float = 1.0  # Rule confidence


class FuzzyLogicLayer(nn.Module):
    """
    Fuzzy logic layer with differentiable operations.

    Implements fuzzy AND, OR, NOT operations that are differentiable.
    """

    def __init__(self, num_features: int, num_rules: int):
        """
        Args:
            num_features: Number of input features
            num_rules: Number of fuzzy rules
        """
        super().__init__()
        self.num_features = num_features
        self.num_rules = num_rules

        # Learnable membership functions (Gaussian)
        self.mu = nn.Parameter(torch.randn(num_rules, num_features))
        self.sigma = nn.Parameter(torch.ones(num_rules, num_features) * 0.5)

        # Rule weights
        self.rule_weights = nn.Parameter(torch.ones(num_rules))

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fuzzy membership degrees.

        Args:
            x: Input tensor [B, F]

        Returns:
            Membership degrees [B, R, F]
        """
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)  # [B, 1, F]
        mu_expanded = self.mu.unsqueeze(0)  # [1, R, F]
        sigma_expanded = self.sigma.unsqueeze(0)  # [1, R, F]

        # Gaussian membership function
        membership = torch.exp(-((x_expanded - mu_expanded) ** 2) / (2 * sigma_expanded ** 2))

        return membership  # [B, R, F]

    def fuzzy_and(self, memberships: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Differentiable fuzzy AND (product t-norm)."""
        return torch.prod(memberships, dim=dim)

    def fuzzy_or(self, memberships: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Differentiable fuzzy OR (probabilistic sum)."""
        return 1 - torch.prod(1 - memberships, dim=dim)

    def fuzzy_not(self, membership: torch.Tensor) -> torch.Tensor:
        """Differentiable fuzzy NOT."""
        return 1 - membership

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fuzzy logic layer.

        Args:
            x: Input tensor [B, F]

        Returns:
            Output tensor [B, R]
        """
        # Compute memberships
        memberships = self.membership(x)  # [B, R, F]

        # Apply fuzzy AND across features for each rule
        rule_activations = self.fuzzy_and(memberships, dim=-1)  # [B, R]

        # Weight rules
        weighted_rules = rule_activations * self.rule_weights.unsqueeze(0)  # [B, R]

        return weighted_rules

    def extract_rules(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Extract human-readable rules from learned parameters.

        Args:
            feature_names: Names of features

        Returns:
            List of rule strings
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.num_features)]

        rules = []

        for rule_idx in range(self.num_rules):
            conditions = []

            for feat_idx in range(self.num_features):
                mu = self.mu[rule_idx, feat_idx].item()
                sigma = self.sigma[rule_idx, feat_idx].item()

                # Create condition based on membership function
                conditions.append(f"{feature_names[feat_idx]} ≈ {mu:.2f} (σ={sigma:.2f})")

            weight = self.rule_weights[rule_idx].item()
            rule_str = f"IF {' AND '.join(conditions)} THEN ... (weight={weight:.3f})"
            rules.append(rule_str)

        return rules


class SymbolicReasoner(nn.Module):
    """
    Symbolic reasoning layer that integrates with PGU.

    Performs logical reasoning and constraint checking.
    """

    def __init__(self,
                 knowledge_base: Optional[List[LogicRule]] = None,
                 use_solver: bool = True):
        """
        Args:
            knowledge_base: Initial knowledge base of rules
            use_solver: Use Z3 solver for verification
        """
        super().__init__()
        self.knowledge_base = knowledge_base or []
        self.use_solver = use_solver

        if use_solver:
            self.pgu = PGUCache(mode="inference")

    def add_rule(self, rule: LogicRule):
        """Add rule to knowledge base."""
        self.knowledge_base.append(rule)

    def verify_constraint(self, constraint: str) -> bool:
        """
        Verify logical constraint using Z3 solver.

        Args:
            constraint: Logical constraint string

        Returns:
            True if satisfiable
        """
        if not self.use_solver:
            return True

        result = self.pgu.check(constraint)
        return result == "sat"

    def apply_rules(self, facts: Dict[str, float]) -> Dict[str, float]:
        """
        Apply rules to derive new facts.

        Args:
            facts: Dictionary of known facts

        Returns:
            Dictionary of derived facts
        """
        derived = facts.copy()

        # Simple forward chaining
        for rule in self.knowledge_base:
            # Check if premise is satisfied
            try:
                # Evaluate premise with current facts
                premise_satisfied = self._evaluate_premise(rule.premise, derived)

                if premise_satisfied:
                    # Apply conclusion
                    self._apply_conclusion(rule.conclusion, derived, rule.confidence)
            except:
                continue

        return derived

    def _evaluate_premise(self, premise: str, facts: Dict[str, float]) -> bool:
        """Evaluate if premise is satisfied given facts."""
        # Simple evaluation - can be extended
        # Replace variables with values
        expr = premise
        for var, val in facts.items():
            expr = expr.replace(var, str(val))

        try:
            # Evaluate boolean expression
            result = eval(expr.replace("AND", "and").replace("OR", "or"))
            return bool(result)
        except:
            return False

    def _apply_conclusion(self, conclusion: str, facts: Dict[str, float], confidence: float):
        """Apply conclusion to facts."""
        # Simple pattern: "X = Y" or "X > Y"
        match = re.match(r"(\w+)\s*=\s*(.+)", conclusion)
        if match:
            var = match.group(1)
            expr = match.group(2)

            # Replace variables in expression
            for fact_var, fact_val in facts.items():
                expr = expr.replace(fact_var, str(fact_val))

            try:
                value = eval(expr) * confidence
                facts[var] = value
            except:
                pass

    def forward(self, x: torch.Tensor, facts: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with symbolic reasoning.

        Args:
            x: Input tensor
            facts: Optional dictionary of symbolic facts

        Returns:
            (output_tensor, derived_facts)
        """
        # For batch processing, apply reasoning to each sample
        batch_size = x.shape[0]
        outputs = []
        all_derived_facts = []

        for i in range(batch_size):
            sample_facts = {}

            if facts is not None:
                # Extract facts for this sample
                for key, val in facts.items():
                    if isinstance(val, torch.Tensor):
                        sample_facts[key] = val[i].item()
                    else:
                        sample_facts[key] = val

            # Apply rules
            derived = self.apply_rules(sample_facts)
            all_derived_facts.append(derived)

            # Convert derived facts to tensor (placeholder)
            # In practice, this would depend on the specific task
            outputs.append(x[i])

        output_tensor = torch.stack(outputs)

        return output_tensor, all_derived_facts


class NeuralSymbolicModule(nn.Module):
    """
    Combined neural-symbolic module.

    Integrates neural networks with symbolic reasoning.
    """

    def __init__(self,
                 neural_encoder: nn.Module,
                 num_fuzzy_rules: int = 10,
                 knowledge_base: Optional[List[LogicRule]] = None,
                 use_symbolic_reasoning: bool = True):
        """
        Args:
            neural_encoder: Neural network for feature extraction
            num_fuzzy_rules: Number of fuzzy logic rules
            knowledge_base: Symbolic knowledge base
            use_symbolic_reasoning: Enable symbolic reasoning
        """
        super().__init__()

        self.neural_encoder = neural_encoder
        self.use_symbolic_reasoning = use_symbolic_reasoning

        # Get feature dimension from encoder
        # Assuming encoder outputs 2D tensor
        self.feature_dim = self._get_feature_dim()

        # Fuzzy logic layer
        self.fuzzy_layer = FuzzyLogicLayer(
            num_features=self.feature_dim,
            num_rules=num_fuzzy_rules
        )

        # Symbolic reasoner
        if use_symbolic_reasoning:
            self.symbolic_reasoner = SymbolicReasoner(knowledge_base=knowledge_base)
        else:
            self.symbolic_reasoner = None

        # Output layer
        self.output_layer = nn.Linear(num_fuzzy_rules, self.feature_dim)

    def _get_feature_dim(self) -> int:
        """Infer feature dimension from encoder."""
        dummy_input = torch.randn(1, 10)  # Assume 10-dim input
        try:
            dummy_output = self.neural_encoder(dummy_input)
            return dummy_output.shape[-1]
        except:
            return 64  # Default

    def forward(self,
                x: torch.Tensor,
                return_explanations: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through neural-symbolic module.

        Args:
            x: Input tensor
            return_explanations: Return symbolic explanations

        Returns:
            (output, explanations_dict)
        """
        # Neural encoding
        neural_features = self.neural_encoder(x)

        # Fuzzy logic reasoning
        fuzzy_activations = self.fuzzy_layer(neural_features)

        # Output
        output = self.output_layer(fuzzy_activations)

        # Symbolic reasoning (optional)
        explanations = None
        if return_explanations and self.symbolic_reasoner is not None:
            # Extract facts from neural features
            facts = {f"feat_{i}": neural_features[0, i].item()
                    for i in range(min(self.feature_dim, 10))}

            # Apply symbolic reasoning
            _, derived_facts = self.symbolic_reasoner(neural_features, facts)

            explanations = {
                "fuzzy_rules": self.fuzzy_layer.extract_rules(),
                "derived_facts": derived_facts,
                "rule_activations": fuzzy_activations.detach().cpu().numpy().tolist()
            }

        return output, explanations

    def explain_prediction(self, x: torch.Tensor, top_k: int = 3) -> str:
        """
        Generate human-readable explanation for prediction.

        Args:
            x: Input sample
            top_k: Number of top rules to include

        Returns:
            Explanation string
        """
        with torch.no_grad():
            output, explanations = self.forward(x.unsqueeze(0), return_explanations=True)

            if explanations is None:
                return "No explanation available"

            # Get top activated rules
            activations = torch.tensor(explanations["rule_activations"][0])
            top_indices = torch.topk(activations, min(top_k, len(activations))).indices

            explanation_parts = ["Prediction based on:"]
            for idx in top_indices:
                rule = explanations["fuzzy_rules"][idx]
                activation = activations[idx].item()
                explanation_parts.append(f"  - {rule} (activation: {activation:.3f})")

            return "\n".join(explanation_parts)


class SymbolicKnowledgeGraph:
    """
    Knowledge graph for storing and querying symbolic knowledge.
    """

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.entities: Set[str] = set()
        self.relations: Dict[Tuple[str, str, str], float] = {}  # (subj, rel, obj) -> weight

    def add_triple(self, subject: str, relation: str, obj: str, weight: float = 1.0):
        """Add triple to knowledge graph."""
        self.entities.add(subject)
        self.entities.add(obj)
        self.relations[(subject, relation, obj)] = weight

    def query(self, subject: Optional[str] = None,
              relation: Optional[str] = None,
              obj: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """
        Query knowledge graph.

        Args:
            subject: Subject filter (None = any)
            relation: Relation filter (None = any)
            obj: Object filter (None = any)

        Returns:
            List of matching triples with weights
        """
        results = []

        for (s, r, o), weight in self.relations.items():
            if subject is not None and s != subject:
                continue
            if relation is not None and r != relation:
                continue
            if obj is not None and o != obj:
                continue

            results.append((s, r, o, weight))

        return results

    def get_neighbors(self, entity: str) -> List[Tuple[str, str, float]]:
        """Get all neighbors of an entity."""
        neighbors = []

        for (s, r, o), weight in self.relations.items():
            if s == entity:
                neighbors.append((r, o, weight))
            elif o == entity:
                neighbors.append((r + "_inv", s, weight))

        return neighbors


if __name__ == "__main__":
    # Demo neural-symbolic integration
    print("=== Neural-Symbolic Integration Demo ===\n")

    # Test fuzzy logic layer
    print("1. Fuzzy Logic Layer")
    fuzzy_layer = FuzzyLogicLayer(num_features=3, num_rules=5)
    x = torch.randn(2, 3)  # 2 samples, 3 features

    output = fuzzy_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    rules = fuzzy_layer.extract_rules(feature_names=["temperature", "pressure", "humidity"])
    print(f"\nExtracted rules:")
    for i, rule in enumerate(rules[:3], 1):
        print(f"  {i}. {rule}")

    # Test symbolic reasoner
    print("\n2. Symbolic Reasoner")
    kb = [
        LogicRule(premise="X > 0 AND Y > 0", conclusion="Z = X + Y", confidence=1.0),
        LogicRule(premise="X > 10", conclusion="category = 'high'", confidence=0.9)
    ]

    reasoner = SymbolicReasoner(knowledge_base=kb, use_solver=False)
    facts = {"X": 5.0, "Y": 3.0}
    derived = reasoner.apply_rules(facts)
    print(f"Input facts: {facts}")
    print(f"Derived facts: {derived}")

    # Test knowledge graph
    print("\n3. Knowledge Graph")
    kg = SymbolicKnowledgeGraph()
    kg.add_triple("Alice", "knows", "Bob", weight=0.9)
    kg.add_triple("Bob", "works_at", "Company_X", weight=1.0)
    kg.add_triple("Alice", "lives_in", "NYC", weight=1.0)

    results = kg.query(subject="Alice")
    print(f"Query results for subject='Alice':")
    for s, r, o, w in results:
        print(f"  ({s}, {r}, {o}) [weight={w}]")

    neighbors = kg.get_neighbors("Bob")
    print(f"\nNeighbors of 'Bob':")
    for r, e, w in neighbors:
        print(f"  {r} -> {e} [weight={w}]")

    print("\nNeural-symbolic demo complete!")
