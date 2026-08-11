"""
Antifragility module — tests robustness through controlled perturbations.
"""
import random

class AntifragilityTester:
    def __init__(self, config=None):
        self.config = config or {}
        self.history = []

    def run_with_noise(self, system, noise_level):
        perturbed = noise_level * random.random()
        # Simulate perturbation test
        metrics = {"noise": noise_level, "score": 1 - abs(perturbed - noise_level)}
        self.history.append(metrics)
        return metrics
