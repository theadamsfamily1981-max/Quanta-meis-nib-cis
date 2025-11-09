"""
Simple import test to confirm scaffolding integrity.
"""
from core_system import tfan, nib_loop, meis_orchestrator, antifragility, multimodal_adapter

def test_imports():
    print("TFAN:", tfan.TFAN)
    print("NIBLoop:", nib_loop.NIBLoop)
    print("MEISOrchestrator:", meis_orchestrator.MEISOrchestrator)
    print("AntifragilityTester:", antifragility.AntifragilityTester)
    print("Adapters:", multimodal_adapter.TextAdapter, multimodal_adapter.ImageAdapter)

if __name__ == "__main__":
    test_imports()
    print("✅ Core system scaffolding imported successfully.")
