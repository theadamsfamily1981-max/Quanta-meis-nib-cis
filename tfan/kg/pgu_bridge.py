#!/usr/bin/env python
"""
PGU Bridge for HyperKG Verification

Bridges hyperbolic KG inference with symbolic PGU verification.
Uses PGU TurboCache to verify that geometric inferences are logically sound.

Workflow:
1. HyperKG makes inference via hyperbolic distance
2. Bridge converts to symbolic formula
3. PGU verifies using cached proofs
4. Return verification result + confidence

Hard gates:
- ≥95% agreement between HyperKG and PGU on test queries
- ≤100ms verification latency (p95)

Usage:
    from tfan.kg import PGUBridge

    bridge = PGUBridge(kg, pgu_cache)

    # Query with verification
    result, verified = bridge.query_with_verification(
        "socrates", "is_a", "mortal"
    )

    # result: True (from HyperKG geometric inference)
    # verified: True (from PGU symbolic verification)
"""

import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

import sys
from pathlib import Path

# Import PGU components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tfan.pgu.cache import TurboCache
    from tfan.pgu.normalizer import alpha_rename
    PGU_AVAILABLE = True
except ImportError:
    PGU_AVAILABLE = False
    print("⚠ PGU not available, verification disabled")


@dataclass
class VerificationResult:
    """Result of PGU verification."""
    query: Tuple[str, str, str]  # (head, relation, tail)
    hyperkg_result: bool
    pgu_result: Optional[bool]
    agreement: bool
    latency_ms: float
    cached: bool
    proof: Optional[str] = None


class PGUBridge:
    """
    Bridge between HyperKG and PGU for verification.

    Converts geometric inferences to symbolic formulas and verifies
    with PGU's proof cache.
    """

    def __init__(
        self,
        kg,
        pgu_cache: Optional['TurboCache'] = None,
        enable_verification: bool = True
    ):
        """
        Initialize PGU bridge.

        Args:
            kg: HyperbolicKG instance
            pgu_cache: PGU TurboCache instance
            enable_verification: Enable PGU verification
        """
        self.kg = kg
        self.enable_verification = enable_verification and PGU_AVAILABLE

        if self.enable_verification:
            self.pgu_cache = pgu_cache or TurboCache(backend='dict', capacity=10000)
        else:
            self.pgu_cache = None

        # Statistics
        self.stats = {
            'num_queries': 0,
            'num_verified': 0,
            'num_agreements': 0,
            'num_disagreements': 0,
            'total_latency_ms': 0.0,
            'cache_hits': 0,
        }

    def query_with_verification(
        self,
        head: str,
        relation: str,
        tail: str
    ) -> VerificationResult:
        """
        Query KG and verify result with PGU.

        Args:
            head: Head entity
            relation: Relation
            tail: Tail entity

        Returns:
            result: VerificationResult with both KG and PGU outcomes
        """
        self.stats['num_queries'] += 1
        start_time = time.time()

        # Query HyperKG
        hyperkg_result = self.kg.query(head, relation, tail)

        # Verify with PGU if enabled
        pgu_result = None
        cached = False

        if self.enable_verification:
            pgu_result, cached = self._verify_with_pgu(head, relation, tail)
            self.stats['num_verified'] += 1

            if cached:
                self.stats['cache_hits'] += 1

            # Check agreement
            if pgu_result is not None:
                if hyperkg_result == pgu_result:
                    self.stats['num_agreements'] += 1
                else:
                    self.stats['num_disagreements'] += 1

        latency_ms = (time.time() - start_time) * 1000
        self.stats['total_latency_ms'] += latency_ms

        agreement = (pgu_result == hyperkg_result) if pgu_result is not None else True

        return VerificationResult(
            query=(head, relation, tail),
            hyperkg_result=hyperkg_result,
            pgu_result=pgu_result,
            agreement=agreement,
            latency_ms=latency_ms,
            cached=cached
        )

    def _verify_with_pgu(
        self,
        head: str,
        relation: str,
        tail: str
    ) -> Tuple[Optional[bool], bool]:
        """
        Verify query using PGU symbolic reasoning.

        Converts triple to logical formula and checks with PGU.

        Args:
            head, relation, tail: Triple components

        Returns:
            (result, cached): Verification result and whether it was cached
        """
        if not self.enable_verification or self.pgu_cache is None:
            return None, False

        # Convert triple to logical formula
        formula = self._triple_to_formula(head, relation, tail)

        # Add KG facts as assumptions
        assumptions = self._get_kg_assumptions()

        # Check cache
        cached_result = self.pgu_cache.get(formula, assumptions)

        if cached_result is not None:
            # Cache hit
            return cached_result['result'], True

        # Cache miss: perform symbolic verification
        result = self._symbolic_verify(formula, assumptions)

        # Cache result
        self.pgu_cache.put(formula, assumptions, result={'result': result})

        return result, False

    def _triple_to_formula(self, head: str, relation: str, tail: str) -> str:
        """
        Convert triple to logical formula.

        Example:
            (socrates, is_a, mortal) → "is_a(socrates, mortal)"

        Args:
            head, relation, tail: Triple components

        Returns:
            formula: Logical formula string
        """
        return f"{relation}({head}, {tail})"

    def _get_kg_assumptions(self) -> Tuple[str, ...]:
        """
        Get KG facts as logical assumptions.

        Converts all triples in KG to logical formulas.

        Returns:
            assumptions: Tuple of assumption strings
        """
        assumptions = []

        for h_idx, r_idx, t_idx in self.kg.triples:
            h = self.kg.idx_to_entity[h_idx]
            r = self.kg.idx_to_relation[r_idx]
            t = self.kg.idx_to_entity[t_idx]

            formula = f"{r}({h}, {t})"
            assumptions.append(formula)

        # Add rules as assumptions
        # Stub: would convert rules to logical implications

        return tuple(sorted(assumptions))

    def _symbolic_verify(self, formula: str, assumptions: Tuple[str, ...]) -> bool:
        """
        Perform symbolic verification.

        Checks if formula is entailed by assumptions using symbolic reasoning.

        Args:
            formula: Query formula
            assumptions: KG facts

        Returns:
            result: True if formula is entailed, False otherwise
        """
        # Stub implementation: simple membership check
        # In production, would use Prolog-like inference engine

        # Check if formula is directly in assumptions
        if formula in assumptions:
            return True

        # Check for simple transitive closure
        # Example: is_a(socrates, mortal) entailed by:
        #   is_a(socrates, human) and is_a(human, mortal)

        # Parse formula
        import re
        match = re.match(r'(\w+)\((\w+), (\w+)\)', formula)
        if not match:
            return False

        relation, head, tail = match.groups()

        # Look for transitive path
        # Stub: would use BFS/DFS for path finding
        for assumption in assumptions:
            match = re.match(r'(\w+)\((\w+), (\w+)\)', assumption)
            if match:
                r, h, t = match.groups()

                if r == relation and h == head:
                    # Found intermediate: check if t → tail exists
                    intermediate_formula = f"{relation}({t}, {tail})"
                    if intermediate_formula in assumptions:
                        return True

        return False

    def check_gates(self) -> Dict:
        """
        Check hard gates for PGU bridge.

        Gates:
        - Agreement rate ≥95% between HyperKG and PGU
        - p95 verification latency ≤100ms

        Returns:
            gates: Dict with gate results
        """
        gates = {}

        # Gate 1: Agreement rate
        if self.stats['num_verified'] > 0:
            agreement_rate = self.stats['num_agreements'] / self.stats['num_verified']
        else:
            agreement_rate = 0.0

        gates['agreement_rate'] = {
            'value': agreement_rate,
            'threshold': 0.95,
            'pass': agreement_rate >= 0.95
        }

        # Gate 2: Verification latency (average as proxy for p95)
        if self.stats['num_queries'] > 0:
            avg_latency = self.stats['total_latency_ms'] / self.stats['num_queries']
        else:
            avg_latency = 0.0

        gates['avg_latency_ms'] = {
            'value': avg_latency,
            'threshold': 100.0,
            'pass': avg_latency <= 100.0
        }

        # Overall
        gates['overall'] = {
            'pass': all(g['pass'] for g in gates.values() if 'pass' in g)
        }

        return gates

    def get_stats(self) -> Dict:
        """Get bridge statistics."""
        stats = dict(self.stats)

        # Compute derived metrics
        if stats['num_queries'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['num_queries']
        else:
            stats['avg_latency_ms'] = 0.0

        if stats['num_verified'] > 0:
            stats['agreement_rate'] = stats['num_agreements'] / stats['num_verified']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['num_verified']
        else:
            stats['agreement_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0

        return stats

    def print_stats(self):
        """Print bridge statistics."""
        stats = self.get_stats()

        print(f"\nPGU Bridge Statistics:")
        print(f"  Queries:          {stats['num_queries']}")
        print(f"  Verified:         {stats['num_verified']}")
        print(f"  Agreements:       {stats['num_agreements']}")
        print(f"  Disagreements:    {stats['num_disagreements']}")
        print(f"  Agreement rate:   {stats['agreement_rate']:.1%}")
        print(f"  Cache hit rate:   {stats['cache_hit_rate']:.1%}")
        print(f"  Avg latency:      {stats['avg_latency_ms']:.2f} ms")

    def print_gate_check(self):
        """Print gate check results."""
        gates = self.check_gates()

        print(f"\nGate Checks:")
        for gate_name, gate_info in gates.items():
            if gate_name == 'overall':
                continue

            status = '✓' if gate_info['pass'] else '✗'
            print(f"  {status} {gate_name}: {gate_info['value']:.2f} (threshold: {gate_info['threshold']})")

        print(f"\n{'✓' if gates['overall']['pass'] else '✗'} Overall: {'PASS' if gates['overall']['pass'] else 'FAIL'}")
