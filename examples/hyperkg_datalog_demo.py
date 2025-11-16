#!/usr/bin/env python
"""
HyperKG + Datalog + PGU Demo

Demonstrates integration of:
1. Hyperbolic KG embeddings (geometric reasoning)
2. Datalog rule compilation (symbolic reasoning)
3. PGU verification (proof checking)

Example domain: Family relationships with transitivity.

Usage:
    python examples/hyperkg_datalog_demo.py
"""

import sys
from pathlib import Path

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.kg import HyperbolicKG, DatalogCompiler, PGUBridge


def create_family_kg() -> HyperbolicKG:
    """Create a simple family KG."""
    kg = HyperbolicKG(embedding_dim=32, distance_threshold=0.8)

    # Add parent relations
    kg.add_triple("alice", "parent", "bob")
    kg.add_triple("bob", "parent", "charlie")
    kg.add_triple("charlie", "parent", "diana")

    kg.add_triple("eve", "parent", "frank")
    kg.add_triple("frank", "parent", "george")

    # Add sibling relations
    kg.add_triple("bob", "sibling", "brenda")
    kg.add_triple("brenda", "sibling", "bob")

    # Add is_a hierarchy
    kg.add_triple("alice", "is_a", "human")
    kg.add_triple("bob", "is_a", "human")
    kg.add_triple("human", "is_a", "mortal")

    return kg


def add_datalog_rules(compiler: DatalogCompiler):
    """Add Datalog rules for inference."""

    # Rule 1: Transitivity of parent → ancestor
    compiler.add_rule("ancestor(?X, ?Z) :- parent(?X, ?Y), parent(?Y, ?Z)")

    # Rule 2: Symmetry of sibling
    compiler.add_rule("sibling(?X, ?Y) :- sibling(?Y, ?X)")

    # Rule 3: Property inheritance via is_a
    compiler.add_rule("is_a(?X, ?Z) :- is_a(?X, ?Y), is_a(?Y, ?Z)")


def main():
    print("=" * 60)
    print("HyperKG + Datalog + PGU Integration Demo")
    print("=" * 60)

    # Create KG
    print("\n1. Creating family knowledge graph...")
    kg = create_family_kg()

    stats = kg.get_stats()
    print(f"   Entities:  {stats['num_entities']}")
    print(f"   Relations: {stats['num_relations']}")
    print(f"   Triples:   {stats['num_triples']}")

    # Compile Datalog rules
    print("\n2. Compiling Datalog rules...")
    compiler = DatalogCompiler(kg)
    add_datalog_rules(compiler)

    compiler.print_rules()

    constraints = compiler.compile_rules()
    compiler.print_constraints()

    # Train hyperbolic embeddings
    print("\n3. Training hyperbolic embeddings...")
    kg.train(epochs=100, lr=0.01)

    # Initialize PGU bridge
    print("\n4. Initializing PGU verification bridge...")
    bridge = PGUBridge(kg, enable_verification=True)

    # Test queries
    print("\n5. Testing queries with verification...")

    test_queries = [
        # Direct facts (should be True)
        ("alice", "parent", "bob"),
        ("bob", "parent", "charlie"),

        # Transitive inference (should be True via ancestor rule)
        ("alice", "parent", "charlie"),  # alice → bob → charlie

        # Multi-hop transitivity
        ("alice", "parent", "diana"),  # alice → bob → charlie → diana

        # Sibling symmetry
        ("brenda", "sibling", "bob"),  # Explicit
        ("bob", "sibling", "brenda"),  # Should be symmetric

        # is_a transitivity
        ("alice", "is_a", "human"),  # Explicit
        ("alice", "is_a", "mortal"),  # Via human → mortal

        # Negative cases
        ("alice", "parent", "frank"),  # False (different family)
        ("eve", "parent", "charlie"),  # False
    ]

    results = []

    for head, relation, tail in test_queries:
        result = bridge.query_with_verification(head, relation, tail)
        results.append(result)

        status = "✓" if result.hyperkg_result else "✗"
        verified = "✓" if result.agreement else "⚠"

        print(f"\n   Query: {head} --[{relation}]--> {tail}")
        print(f"     HyperKG:  {status} {result.hyperkg_result}")
        print(f"     PGU:      {result.pgu_result}")
        print(f"     Verified: {verified} {result.agreement}")
        print(f"     Latency:  {result.latency_ms:.2f} ms")
        print(f"     Cached:   {result.cached}")

    # Print statistics
    print("\n" + "=" * 60)
    bridge.print_stats()

    # Check gates
    print("\n" + "=" * 60)
    bridge.print_gate_check()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    correct = sum(1 for r in results if r.hyperkg_result)
    total = len(results)

    print(f"\nQueries:")
    print(f"  Total:     {total}")
    print(f"  Correct:   {correct}")
    print(f"  Accuracy:  {correct/total:.1%}")

    gates = bridge.check_gates()
    if gates['overall']['pass']:
        print(f"\n✓ All gates PASSED")
        return 0
    else:
        print(f"\n✗ Some gates FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
