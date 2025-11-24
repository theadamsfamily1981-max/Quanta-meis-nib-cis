"""Proof goal utilities used in the PGU cache tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple
import ast
import operator

_ALLOWED_NODES = {
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.And: lambda *args: all(args),
    ast.Or: lambda *args: any(args),
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _safe_eval(expression: str, substitutions: Mapping[str, float]) -> bool:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
    return _eval_node(tree.body, substitutions)


def _eval_node(node: ast.AST, substitutions: Mapping[str, float]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        try:
            return substitutions[node.id]
        except KeyError as exc:
            raise KeyError(f"missing substitution for {node.id}") from exc
    if isinstance(node, ast.UnaryOp):
        fn = _ALLOWED_OPERATORS[type(node.op)]
        return fn(_eval_node(node.operand, substitutions))
    if isinstance(node, ast.BinOp):
        fn = _ALLOWED_OPERATORS[type(node.op)]
        return fn(_eval_node(node.left, substitutions), _eval_node(node.right, substitutions))
    if isinstance(node, ast.BoolOp):
        fn = _ALLOWED_OPERATORS[type(node.op)]
        return fn(*[_eval_node(v, substitutions) for v in node.values])
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, substitutions)
        result = True
        for comparator, op in zip(node.comparators, node.ops):
            fn = _ALLOWED_OPERATORS[type(op)]
            right = _eval_node(comparator, substitutions)
            if not fn(left, right):
                result = False
                break
            left = right
        return result
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@dataclass(frozen=True)
class CacheKey:
    goal: str
    substitutions: Tuple[Tuple[str, float], ...]

    @classmethod
    def from_goal(cls, goal: str, substitutions: Mapping[str, float]) -> "CacheKey":
        ordered = tuple(sorted((name, float(value)) for name, value in substitutions.items()))
        return cls(goal.strip(), ordered)


class ProofGoalCache:
    """Substitution-aware memoisation structure."""

    def __init__(self) -> None:
        self._store: Dict[CacheKey, bool] = {}

    def get(self, goal: str, substitutions: Mapping[str, float]) -> bool | None:
        return self._store.get(CacheKey.from_goal(goal, substitutions))

    def add(self, goal: str, substitutions: Mapping[str, float], result: bool) -> None:
        key = CacheKey.from_goal(goal, substitutions)
        self._store[key] = bool(result)

    def clear(self) -> None:
        self._store.clear()


class Z3LiteProofChecker:
    """A tiny, deterministic proof checker used by the unit tests.

    It only supports arithmetic expressions with a small subset of comparison and
    boolean operators.  The :class:`ProofGoalCache` can be injected to reuse
    previous results.
    """

    def __init__(self, cache: ProofGoalCache | None = None) -> None:
        self.cache = cache or ProofGoalCache()

    def check(self, goal: str, substitutions: Mapping[str, float]) -> bool:
        cached = self.cache.get(goal, substitutions)
        if cached is not None:
            return cached
        result = bool(_safe_eval(goal, substitutions))
        self.cache.add(goal, substitutions, result)
        return result


__all__ = ["ProofGoalCache", "Z3LiteProofChecker"]
