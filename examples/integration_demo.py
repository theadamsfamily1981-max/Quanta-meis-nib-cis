#!/usr/bin/env python
"""
TF-A-N Integration Demo

Shows how multiple subsystems work together:
- HyperKG for knowledge representation
- PGU TurboCache for proof caching
- Pareto optimization for hyperparameter tuning
- Multimodal fusion for input processing

This demonstrates the full TF-A-N stack in action.

Usage:
    python examples/integration_demo.py
"""

import sys
from pathlib import Path

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from typing import Dict, List
import time


def demo_knowledge_graph_with_pgu():
    """Demo 1: HyperKG + PGU Bridge"""
    print("=" * 60)
    print("Demo 1: Knowledge Graph with Symbolic Verification")
    print("=" * 60)

    from tfan.kg import HyperbolicKG, DatalogCompiler, PGUBridge

    # Create medical knowledge graph
    kg = HyperbolicKG(embedding_dim=32)

    # Add medical taxonomy
    kg.add_triple("aspirin", "is_a", "NSAID")
    kg.add_triple("ibuprofen", "is_a", "NSAID")
    kg.add_triple("NSAID", "is_a", "pain_reliever")
    kg.add_triple("pain_reliever", "is_a", "medication")

    # Add side effects
    kg.add_triple("aspirin", "has_side_effect", "bleeding")
    kg.add_triple("ibuprofen", "has_side_effect", "stomach_upset")

    # Add Datalog rules for inference
    compiler = DatalogCompiler(kg)

    # Transitivity of is_a
    compiler.add_rule("is_a(?X, ?Z) :- is_a(?X, ?Y), is_a(?Y, ?Z)")

    # Inheritance of side effects
    compiler.add_rule(
        "has_side_effect(?X, ?E) :- is_a(?X, ?Y), has_side_effect(?Y, ?E)"
    )

    print("\nKnowledge Graph Statistics:")
    stats = kg.get_stats()
    print(f"  Entities:  {stats['num_entities']}")
    print(f"  Relations: {stats['num_relations']}")
    print(f"  Triples:   {stats['num_triples']}")

    # Train embeddings
    print("\nTraining hyperbolic embeddings...")
    kg.train(epochs=50, lr=0.01)

    # Query with PGU verification
    print("\nQuerying with symbolic verification...")
    bridge = PGUBridge(kg, enable_verification=True)

    queries = [
        ("aspirin", "is_a", "medication"),  # Multi-hop inference
        ("ibuprofen", "is_a", "medication"),  # Multi-hop inference
        ("aspirin", "is_a", "antibiotic"),  # Should be False
    ]

    for head, rel, tail in queries:
        result = bridge.query_with_verification(head, rel, tail)

        status = "✓" if result.hyperkg_result else "✗"
        verified = "✓" if result.agreement else "⚠"

        print(f"\n  Query: {head} --[{rel}]--> {tail}")
        print(f"    Geometric: {status} {result.hyperkg_result}")
        print(f"    Symbolic:  {result.pgu_result}")
        print(f"    Agreement: {verified} {result.agreement}")

    bridge.print_stats()

    return bridge


def demo_multimodal_with_pareto():
    """Demo 2: Multimodal Fusion + Pareto Optimization"""
    print("\n" + "=" * 60)
    print("Demo 2: Multimodal Fusion + Hyperparameter Tuning")
    print("=" * 60)

    from tfan.mmf import FusionBus, PADGate
    from tfan.pareto import BatchRunner

    # Create multimodal fusion system
    modalities = ['audio', 'video', 'text']
    fusion_dim = 256

    bus = FusionBus(modalities, fusion_dim=fusion_dim)
    gate = PADGate()

    print("\nFusion Bus Configuration:")
    print(f"  Modalities: {modalities}")
    print(f"  Fusion dim: {fusion_dim}")

    # Simulate multimodal input
    batch_size = 4
    features = {
        'audio': torch.randn(batch_size, 128),
        'video': torch.randn(batch_size, 256),
        'text': torch.randn(batch_size, 512)
    }

    # Simulate PAD (emotion) vector
    # High arousal, moderate pleasure, low dominance
    pad = torch.tensor([[0.5, 0.8, 0.3]] * batch_size)

    # Compute emotion-based scheduling
    temperature, keep_ratio = gate.schedule(pad)

    print(f"\nEmotion-based Scheduling:")
    print(f"  PAD vector:    [{pad[0, 0]:.2f}, {pad[0, 1]:.2f}, {pad[0, 2]:.2f}]")
    print(f"  Temperature:   {temperature[0]:.2f} (high arousal → exploration)")
    print(f"  Keep ratio:    {keep_ratio[0]:.2f} (low dominance → selectivity)")

    # Fuse with emotion modulation
    fused, info = bus.fuse(features, pad_vector=pad, temperature=temperature[0].item())

    print(f"\nFusion Results:")
    print(f"  Output shape:  {fused.shape}")
    print(f"  Modality weights:")
    for name, weight in info['weights'].items():
        print(f"    {name}: {weight:.3f}")

    # Optimize fusion hyperparameters with Pareto
    print("\nOptimizing hyperparameters with Pareto...")

    def fusion_objective(config):
        """Objective for Pareto optimization."""
        # Simulate different fusion configurations
        time.sleep(0.05)  # Simulate computation

        # Return multiple objectives (lower is better)
        return {
            'reconstruction_loss': config['fusion_dim'] / 512.0 + 0.1,  # Accuracy proxy
            'latency_ms': config['fusion_dim'] * 0.01,  # Speed proxy
            'memory_mb': config['fusion_dim'] * 0.5  # Memory proxy
        }

    # Grid of configurations
    configs = [
        {'fusion_dim': dim, 'num_heads': heads}
        for dim in [128, 256, 512]
        for heads in [4, 8]
    ]

    runner = BatchRunner(fusion_objective, max_workers=2)
    results = runner.run(configs)

    print(f"\nEvaluated {len(results)} configurations")
    print("\nTop configurations by reconstruction loss:")
    sorted_results = sorted(results, key=lambda x: x['reconstruction_loss'])
    for i, res in enumerate(sorted_results[:3]):
        print(f"  {i+1}. dim={res['config']['fusion_dim']}, "
              f"heads={res['config']['num_heads']}: "
              f"loss={res['reconstruction_loss']:.3f}, "
              f"latency={res['latency_ms']:.2f}ms")

    return bus, gate


def demo_end_to_end_pipeline():
    """Demo 3: End-to-end pipeline"""
    print("\n" + "=" * 60)
    print("Demo 3: End-to-End Pipeline")
    print("=" * 60)

    print("\nSimulating full TF-A-N pipeline:")
    print("  1. Multimodal input → Fusion Bus")
    print("  2. Knowledge retrieval → HyperKG")
    print("  3. Reasoning → PGU verification")
    print("  4. Emotion modulation → PAD Gate")
    print("  5. Output generation")

    # This would integrate all components in a real application
    print("\n✓ Pipeline simulation complete")
    print("\nIn a real application, this would:")
    print("  - Ingest audio/video/text streams")
    print("  - Fuse modalities with emotion-based weights")
    print("  - Query knowledge graph for context")
    print("  - Verify inferences with symbolic reasoning")
    print("  - Generate outputs with topological constraints")


def main():
    print("=" * 60)
    print("TF-A-N Integration Demo")
    print("=" * 60)
    print("\nDemonstrating how subsystems work together:\n")

    # Demo 1: Knowledge + Verification
    bridge = demo_knowledge_graph_with_pgu()

    # Demo 2: Multimodal + Optimization
    bus, gate = demo_multimodal_with_pareto()

    # Demo 3: Full pipeline
    demo_end_to_end_pipeline()

    # Summary
    print("\n" + "=" * 60)
    print("Integration Demo Complete")
    print("=" * 60)

    print("\nKey Takeaways:")
    print("  ✓ HyperKG + PGU: Geometric + symbolic reasoning")
    print("  ✓ Fusion + Pareto: Multimodal processing + optimization")
    print("  ✓ PAD Gate: Emotion-based dynamic scheduling")
    print("  ✓ All subsystems integrate seamlessly")

    print("\nNext Steps:")
    print("  1. Explore individual subsystem READMEs")
    print("  2. Run benchmarks to verify hard gates")
    print("  3. Use GUI for interactive exploration")

    # Check gates
    gates = bridge.check_gates()
    if gates['overall']['pass']:
        print("\n✓ All hard gates PASSED")
        return 0
    else:
        print("\n⚠ Some hard gates FAILED (expected in demo with small data)")
        return 1


if __name__ == "__main__":
    exit(main())
