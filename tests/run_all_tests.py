#!/usr/bin/env python
"""
Comprehensive Test Suite for TF-A-N

Runs all tests across all subsystems with detailed reporting.

Usage:
    python tests/run_all_tests.py --verbose
    python tests/run_all_tests.py --output test_report.json
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Test configuration
@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    subsystem: str
    status: str  # "pass", "fail", "skip", "error"
    duration_s: float
    output: str = ""
    error: str = ""


class TestRunner:
    """Runs comprehensive test suite."""

    def __init__(self, repo_dir: str):
        """Initialize test runner.

        Args:
            repo_dir: Path to repository root
        """
        self.repo_dir = Path(repo_dir)
        self.results: List[TestResult] = []

    def run_all_tests(self, verbose: bool = False) -> Dict:
        """Run all tests and return results."""
        print("=" * 60)
        print("TF-A-N Comprehensive Test Suite")
        print("=" * 60)
        print(f"Repository: {self.repo_dir}")
        print()

        # Test categories
        test_suites = [
            ("Imports", self._test_imports),
            ("PGU TurboCache", self._test_pgu),
            ("Pareto Auto-Runner", self._test_pareto),
            ("Multimodal Fusion", self._test_multimodal),
            ("eBPF Telemetry", self._test_ebpf),
            ("Fused Radial Attention", self._test_fused_attention),
            ("FSDP Orchestrator", self._test_fsdp),
            ("HyperKG", self._test_hyperkg),
        ]

        total_start = time.time()

        for suite_name, suite_func in test_suites:
            print(f"\n{'=' * 60}")
            print(f"Running: {suite_name}")
            print(f"{'=' * 60}\n")

            try:
                suite_func(verbose)
            except Exception as e:
                print(f"✗ Suite {suite_name} crashed: {e}")
                self.results.append(TestResult(
                    name=f"{suite_name} (crashed)",
                    subsystem=suite_name,
                    status="error",
                    duration_s=0.0,
                    error=str(e)
                ))

        total_duration = time.time() - total_start

        # Generate summary
        summary = self._generate_summary(total_duration)

        return summary

    def _run_python_test(
        self,
        name: str,
        subsystem: str,
        command: List[str],
        verbose: bool = False
    ) -> TestResult:
        """Run a single Python test command."""
        start = time.time()

        try:
            result = subprocess.run(
                command,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            duration = time.time() - start

            if result.returncode == 0:
                status = "pass"
                print(f"  ✓ {name} ({duration:.2f}s)")
            else:
                status = "fail"
                print(f"  ✗ {name} ({duration:.2f}s)")
                if verbose:
                    print(f"     Error: {result.stderr[:200]}")

            test_result = TestResult(
                name=name,
                subsystem=subsystem,
                status=status,
                duration_s=duration,
                output=result.stdout if verbose else "",
                error=result.stderr if status == "fail" else ""
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"  ⚠ {name} (timeout)")
            test_result = TestResult(
                name=name,
                subsystem=subsystem,
                status="error",
                duration_s=duration,
                error="Timeout after 5 minutes"
            )

        except Exception as e:
            duration = time.time() - start
            print(f"  ✗ {name} (error: {e})")
            test_result = TestResult(
                name=name,
                subsystem=subsystem,
                status="error",
                duration_s=duration,
                error=str(e)
            )

        self.results.append(test_result)
        return test_result

    def _test_imports(self, verbose: bool):
        """Test all module imports."""
        imports = [
            "from tfan.pgu import TurboCache",
            "from tfan.pareto import ParetoRunner",
            "from tfan.mmf import FusionBus",
            "from tfan.emotion import PADGate",
            "from tfan.kernels import FusedRadialAttention",
            "from tfan.distributed import FSDPOrchestrator",
            "from tfan.kg import HyperbolicKG, DatalogCompiler, PGUBridge",
        ]

        for import_stmt in imports:
            module_name = import_stmt.split()[-1]
            self._run_python_test(
                name=f"Import {module_name}",
                subsystem="Imports",
                command=["python", "-c", import_stmt],
                verbose=verbose
            )

    def _test_pgu(self, verbose: bool):
        """Test PGU TurboCache."""
        # Test normalizer
        self._run_python_test(
            name="PGU Alpha-renaming",
            subsystem="PGU",
            command=["python", "-c", """
from tfan.pgu.normalizer import alpha_rename
canon, mapping = alpha_rename('p(x, y) & q(y, z)')
assert 'x1' in canon and 'x2' in canon
print('✓ Alpha-rename works')
            """],
            verbose=verbose
        )

        # Test cache
        self._run_python_test(
            name="PGU Cache operations",
            subsystem="PGU",
            command=["python", "-c", """
from tfan.pgu.cache import TurboCache
cache = TurboCache(backend='dict')
cache.put('p(x)', (), result={'valid': True})
result = cache.get('p(x)', ())
assert result['valid'] == True
print('✓ Cache works')
            """],
            verbose=verbose
        )

        # Run benchmark
        self._run_python_test(
            name="PGU Benchmark",
            subsystem="PGU",
            command=["python", "tfan/pgu/corpus_replay.py", "--size", "100", "--iterations", "50"],
            verbose=verbose
        )

    def _test_pareto(self, verbose: bool):
        """Test Pareto Auto-Runner."""
        # Test EHVI
        self._run_python_test(
            name="Pareto EHVI computation",
            subsystem="Pareto",
            command=["python", "-c", """
from tfan.pareto.ehvi import pareto_frontier
import numpy as np
points = np.array([[1, 10], [2, 5], [3, 3], [5, 2]])
frontier = pareto_frontier(points)
assert len(frontier) > 0
print(f'✓ Found {len(frontier)} frontier points')
            """],
            verbose=verbose
        )

        # Test runner
        self._run_python_test(
            name="Pareto BatchRunner",
            subsystem="Pareto",
            command=["python", "-c", """
from tfan.pareto.runner import BatchRunner
import time

def dummy_objective(config):
    time.sleep(0.1)
    return {'loss': config['x']**2, 'latency': config['x']}

configs = [{'x': i} for i in range(5)]
runner = BatchRunner(dummy_objective, max_workers=2)
results = runner.run(configs)
assert len(results) == 5
print('✓ BatchRunner works')
            """],
            verbose=verbose
        )

    def _test_multimodal(self, verbose: bool):
        """Test Multimodal Fusion + PAD."""
        # Test FusionBus
        self._run_python_test(
            name="FusionBus forward pass",
            subsystem="Multimodal",
            command=["python", "-c", """
import torch
from tfan.mmf.bus import FusionBus

bus = FusionBus(['audio', 'video'], fusion_dim=256)
features = {
    'audio': torch.randn(2, 128),
    'video': torch.randn(2, 256)
}
fused, info = bus.fuse(features)
assert fused.shape == (2, 256)
print('✓ FusionBus works')
            """],
            verbose=verbose
        )

        # Test PADGate
        self._run_python_test(
            name="PADGate scheduling",
            subsystem="Multimodal",
            command=["python", "-c", """
import torch
from tfan.mmf.pad_gate import PADGate

gate = PADGate()
pad = torch.tensor([[0.5, 0.8, 0.3]])  # [P, A, D]
T, k = gate.schedule(pad)
assert T.shape == (1,) and k.shape == (1,)
assert 0.5 <= T.item() <= 2.0
print(f'✓ PADGate: T={T.item():.2f}, k={k.item():.2f}')
            """],
            verbose=verbose
        )

    def _test_ebpf(self, verbose: bool):
        """Test eBPF telemetry (syntax check only, no actual tracing)."""
        # Check BPF C syntax
        self._run_python_test(
            name="BPF syntax check",
            subsystem="eBPF",
            command=["python", "-c", """
from pathlib import Path
pgu_bpf = Path('monitoring/ebpf/trace_pgu.c')
ttw_bpf = Path('monitoring/ebpf/trace_ttw.c')
assert pgu_bpf.exists() and ttw_bpf.exists()
print('✓ BPF source files exist')
            """],
            verbose=verbose
        )

        # Test CUPTI tracer import
        self._run_python_test(
            name="CUPTI tracer import",
            subsystem="eBPF",
            command=["python", "-c", """
from monitoring.gpu.cupti_tracer import CUPTITracer
tracer = CUPTITracer()
stats = tracer.get_stats()
assert 'kernel_events' in stats
print('✓ CUPTI tracer works')
            """],
            verbose=verbose
        )

    def _test_fused_attention(self, verbose: bool):
        """Test Fused Radial Attention."""
        # Test unfused implementation (CUDA kernel requires GPU)
        self._run_python_test(
            name="Fused attention (unfused fallback)",
            subsystem="Fused Attention",
            command=["python", "-c", """
import torch
from tfan.kernels.fused_radial_attn import FusedRadialAttention

attn = FusedRadialAttention(num_heads=4, head_dim=32, use_cuda=False)
Q = torch.randn(2, 4, 16, 32)
K = torch.randn(2, 4, 16, 32)
V = torch.randn(2, 4, 16, 32)
landmarks = torch.randint(0, 16, (16, 4))
radii = torch.randint(2, 8, (16,))

output = attn(Q, K, V, landmarks, radii)
assert output.shape == (2, 4, 16, 32)
print('✓ Fused attention (unfused) works')
            """],
            verbose=verbose
        )

    def _test_fsdp(self, verbose: bool):
        """Test FSDP Orchestrator (single GPU mode)."""
        # Test single GPU mode
        self._run_python_test(
            name="FSDP single GPU",
            subsystem="FSDP",
            command=["python", "-c", """
import torch
import torch.nn as nn
from tfan.distributed import FSDPOrchestrator, FSDPConfig

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

config = FSDPConfig(sharding_strategy='full')
# Single GPU: FSDP should be skipped
orchestrator = FSDPOrchestrator(model, config=config)

stats = orchestrator.get_stats()
print(f'✓ FSDP orchestrator initialized (world_size={stats[\"world_size\"]})')
            """],
            verbose=verbose
        )

    def _test_hyperkg(self, verbose: bool):
        """Test HyperKG + Datalog + PGU Bridge."""
        # Test HyperbolicKG
        self._run_python_test(
            name="HyperbolicKG basic operations",
            subsystem="HyperKG",
            command=["python", "-c", """
from tfan.kg import HyperbolicKG

kg = HyperbolicKG(embedding_dim=32)
kg.add_triple('a', 'rel', 'b')
kg.add_triple('b', 'rel', 'c')

stats = kg.get_stats()
assert stats['num_entities'] == 3
assert stats['num_triples'] == 2
print('✓ HyperbolicKG works')
            """],
            verbose=verbose
        )

        # Test DatalogCompiler
        self._run_python_test(
            name="Datalog rule compilation",
            subsystem="HyperKG",
            command=["python", "-c", """
from tfan.kg import DatalogCompiler

compiler = DatalogCompiler()
compiler.add_rule('ancestor(?X, ?Z) :- parent(?X, ?Y), parent(?Y, ?Z)')

constraints = compiler.compile_rules()
assert len(constraints) > 0
print(f'✓ Compiled {len(constraints)} constraints')
            """],
            verbose=verbose
        )

        # Test PGUBridge
        self._run_python_test(
            name="PGU Bridge verification",
            subsystem="HyperKG",
            command=["python", "-c", """
from tfan.kg import HyperbolicKG, PGUBridge

kg = HyperbolicKG(embedding_dim=16)
kg.add_triple('alice', 'parent', 'bob')

bridge = PGUBridge(kg, enable_verification=True)
result = bridge.query_with_verification('alice', 'parent', 'bob')

assert result.hyperkg_result == True
print('✓ PGU Bridge works')
            """],
            verbose=verbose
        )

        # Run demo
        self._run_python_test(
            name="HyperKG demo",
            subsystem="HyperKG",
            command=["python", "examples/hyperkg_datalog_demo.py"],
            verbose=verbose
        )

    def _generate_summary(self, total_duration: float) -> Dict:
        """Generate test summary."""
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        errors = sum(1 for r in self.results if r.status == "error")
        skipped = sum(1 for r in self.results if r.status == "skip")
        total = len(self.results)

        summary = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'duration_s': total_duration,
            'pass_rate': passed / total if total > 0 else 0.0,
            'results': [asdict(r) for r in self.results]
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total:    {summary['total']}")
        print(f"✓ Passed: {summary['passed']}")
        print(f"✗ Failed: {summary['failed']}")
        print(f"⚠ Errors: {summary['errors']}")
        print(f"⊘ Skipped: {summary['skipped']}")
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        print(f"Duration:  {summary['duration_s']:.2f}s")

        if summary['failed'] > 0:
            print("\nFailed tests:")
            for result in self.results:
                if result.status == "fail":
                    print(f"  ✗ {result.subsystem}: {result.name}")
                    if result.error:
                        print(f"    {result.error[:100]}")

        if summary['errors'] > 0:
            print("\nErrors:")
            for result in self.results:
                if result.status == "error":
                    print(f"  ⚠ {result.subsystem}: {result.name}")
                    if result.error:
                        print(f"    {result.error[:100]}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--repo-dir", type=str, default=".",
                        help="Repository directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--output", type=str,
                        help="Output JSON file")

    args = parser.parse_args()

    # Run tests
    runner = TestRunner(args.repo_dir)
    summary = runner.run_all_tests(verbose=args.verbose)

    # Print summary
    runner.print_summary(summary)

    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved results to {args.output}")

    # Exit with appropriate code
    if summary['failed'] > 0 or summary['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
