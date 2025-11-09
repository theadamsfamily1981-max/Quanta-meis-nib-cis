"""
MEIS Orchestrator — coordinates modular system execution.
"""
from core_system.utils.event_bus import EventBus

class MEISOrchestrator:
    def __init__(self, modules=None):
        self.modules = modules or []
        self.bus = EventBus()

    def run(self, input_stream):
        results = []
        for module in self.modules:
            result = module.forward(input_stream)
            self.bus.publish("Result", result)
            results.append(result)
        return results

    def add_module(self, module):
        self.modules.append(module)
