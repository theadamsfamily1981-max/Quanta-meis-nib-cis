"""
Knowledge Graph components with hyperbolic embeddings and Datalog reasoning.

Modules:
- hyperbolic_kg: Hyperbolic KG with Poincaré embeddings
- datalog_compiler: Compile Datalog rules to geometric constraints
- pgu_bridge: Verify KG inferences with PGU symbolic reasoning
"""

from .hyperbolic_kg import HyperbolicKG, PoincareEmbedding
from .datalog_compiler import DatalogCompiler, Rule, Atom, Variable, Constant
from .pgu_bridge import PGUBridge, VerificationResult

__all__ = [
    "HyperbolicKG",
    "PoincareEmbedding",
    "DatalogCompiler",
    "Rule",
    "Atom",
    "Variable",
    "Constant",
    "PGUBridge",
    "VerificationResult"
]
