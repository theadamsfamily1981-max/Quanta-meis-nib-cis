"""
Tests for advanced features: MOEA/D, Neural-Symbolic, Compression, Continuous Adaptation.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from tfan.moead import (
    Objective,
    WeightVector,
    MOEADOptimizer,
    MOEADTrainer
)
from tfan.neuro_symbolic import (
    FuzzyLogicLayer,
    SymbolicReasoner,
    LogicRule,
    NeuralSymbolicModule,
    SymbolicKnowledgeGraph
)
from tfan.compression import (
    ModelQuantizer,
    StructuredPruning,
    CompressionPipeline
)
from tfan.continuous_adaptation import (
    ExperienceReplay,
    ElasticWeightConsolidation,
    OnlineMetaLearner,
    ContinualLearningTrainer
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.mark.advanced
class TestMOEAD:
    """Tests for MOEA/D multi-objective optimization."""

    def test_weight_vector_generation(self):
        """Test uniform weight vector generation."""
        weights = WeightVector.generate_uniform(num_objectives=3, num_vectors=50)

        assert weights.shape == (50, 3)
        # Check simplex constraint (sum to 1)
        assert np.allclose(weights.sum(axis=1), 1.0)
        # Check non-negative
        assert np.all(weights >= 0)

    def test_moead_initialization(self):
        """Test MOEA/D optimizer initialization."""
        objectives = [
            Objective(name="loss", minimize=True),
            Objective(name="complexity", minimize=True),
            Objective(name="accuracy", minimize=False)
        ]

        optimizer = MOEADOptimizer(
            objectives=objectives,
            population_size=50,
            num_neighbors=10
        )

        assert optimizer.num_objectives == 3
        assert optimizer.population_size == 50
        assert optimizer.num_neighbors == 10
        assert optimizer.weight_vectors.shape == (50, 3)
        assert optimizer.neighbors.shape == (50, 10)

    def test_tchebycheff_decomposition(self):
        """Test Tchebycheff decomposition."""
        objectives = [Objective(name="f1"), Objective(name="f2")]
        optimizer = MOEADOptimizer(objectives, population_size=10)

        obj_vals = np.array([0.5, 0.3])
        optimizer.reference_point = np.array([0.0, 0.0])

        fitness = optimizer._tchebycheff_decomposition(obj_vals, np.array([0.5, 0.5]))

        assert isinstance(fitness, (float, np.floating))
        assert fitness >= 0

    def test_pareto_front_extraction(self):
        """Test Pareto front extraction."""
        objectives = [Objective(name="f1"), Objective(name="f2")]
        optimizer = MOEADOptimizer(objectives, population_size=5)

        # Create mock population
        optimizer.population = [i for i in range(5)]
        optimizer.objective_values = [
            [1.0, 5.0],  # Dominated
            [2.0, 3.0],  # Pareto-optimal
            [3.0, 2.0],  # Pareto-optimal
            [4.0, 1.0],  # Pareto-optimal
            [2.5, 3.5]   # Dominated
        ]

        pareto_individuals, pareto_vals = optimizer.get_pareto_front()

        # Should have 3 Pareto-optimal solutions
        assert len(pareto_individuals) == 3
        assert len(pareto_vals) == 3


@pytest.mark.advanced
class TestNeuralSymbolic:
    """Tests for neural-symbolic integration."""

    def test_fuzzy_logic_layer(self):
        """Test fuzzy logic layer."""
        layer = FuzzyLogicLayer(num_features=5, num_rules=10)

        x = torch.randn(4, 5)  # 4 samples, 5 features
        output = layer(x)

        assert output.shape == (4, 10)  # 4 samples, 10 rules

    def test_fuzzy_operations(self):
        """Test fuzzy AND, OR, NOT operations."""
        layer = FuzzyLogicLayer(num_features=3, num_rules=5)

        memberships = torch.tensor([[0.8, 0.6, 0.9]])

        # Fuzzy AND (product t-norm)
        and_result = layer.fuzzy_and(memberships)
        assert and_result <= min(0.8, 0.6, 0.9)

        # Fuzzy OR
        or_result = layer.fuzzy_or(memberships)
        assert or_result >= max(0.8, 0.6, 0.9)

        # Fuzzy NOT
        not_result = layer.fuzzy_not(torch.tensor(0.7))
        assert abs(not_result - 0.3) < 0.01

    def test_fuzzy_rule_extraction(self):
        """Test rule extraction from fuzzy layer."""
        layer = FuzzyLogicLayer(num_features=3, num_rules=5)

        rules = layer.extract_rules(feature_names=["temp", "pressure", "humidity"])

        assert len(rules) == 5
        assert all(isinstance(rule, str) for rule in rules)
        assert all("temp" in rule for rule in rules)

    def test_symbolic_reasoner(self):
        """Test symbolic reasoning."""
        kb = [
            LogicRule(premise="X > 0 AND Y > 0", conclusion="Z = X + Y", confidence=1.0)
        ]

        reasoner = SymbolicReasoner(knowledge_base=kb, use_solver=False)

        facts = {"X": 5.0, "Y": 3.0}
        derived = reasoner.apply_rules(facts)

        assert "Z" in derived
        assert abs(derived["Z"] - 8.0) < 0.1

    def test_knowledge_graph(self):
        """Test knowledge graph operations."""
        kg = SymbolicKnowledgeGraph()

        kg.add_triple("Alice", "knows", "Bob", weight=0.9)
        kg.add_triple("Bob", "works_at", "CompanyX", weight=1.0)

        # Query
        results = kg.query(subject="Alice")
        assert len(results) == 1
        assert results[0] == ("Alice", "knows", "Bob", 0.9)

        # Neighbors
        neighbors = kg.get_neighbors("Bob")
        assert len(neighbors) == 2  # knows_inv and works_at


@pytest.mark.advanced
class TestCompression:
    """Tests for model compression."""

    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = SimpleModel()

        quantizer = ModelQuantizer(model, backend="fbgemm")
        quant_model = quantizer.dynamic_quantize()

        # Test forward pass
        x = torch.randn(2, 10)
        output = quant_model(x)

        assert output.shape == (2, 10)

    def test_model_size_measurement(self):
        """Test model size measurement."""
        model = SimpleModel()

        quantizer = ModelQuantizer(model)
        quant_model = quantizer.dynamic_quantize()

        size_stats = quantizer.measure_model_size(quant_model)

        assert "original_size_mb" in size_stats
        assert "quantized_size_mb" in size_stats
        assert "compression_ratio" in size_stats

        # Quantized should be smaller
        assert size_stats["quantized_size_mb"] < size_stats["original_size_mb"]
        assert size_stats["compression_ratio"] > 1.0

    def test_structured_pruning(self):
        """Test structured pruning."""
        model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=10)

        pruner = StructuredPruning(model)

        # Prune first layer
        pruned_layer, mask = pruner.prune_layer(model.fc1, pruning_ratio=0.5)

        # Should keep approximately 50% of neurons
        assert pruned_layer.out_features < model.fc1.out_features
        assert abs(pruned_layer.out_features / model.fc1.out_features - 0.5) < 0.1

    def test_importance_computation(self):
        """Test importance score computation."""
        model = SimpleModel()
        pruner = StructuredPruning(model)

        importance = pruner.compute_importance(model.fc1, method="l1")

        assert importance.shape[0] == model.fc1.out_features
        assert torch.all(importance >= 0)

    def test_compression_pipeline(self):
        """Test full compression pipeline."""
        model = SimpleModel()

        pipeline = CompressionPipeline(model)

        # Compress with pruning and quantization
        compressed_model, stats = pipeline.compress(
            pruning_ratio=0.3,
            quantization_method="dynamic"
        )

        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 1.0


@pytest.mark.advanced
class TestContinuousAdaptation:
    """Tests for continuous adaptation."""

    def test_experience_replay(self):
        """Test experience replay buffer."""
        replay = ExperienceReplay(capacity=100)

        # Add experiences
        for i in range(50):
            replay.add((torch.randn(10), torch.tensor(i)))

        assert len(replay) == 50

        # Sample
        batch = replay.sample(10)
        assert len(batch) == 10

    def test_prioritized_replay(self):
        """Test prioritized experience replay."""
        replay = ExperienceReplay(capacity=100, prioritized=True)

        # Add with different priorities
        for i in range(50):
            priority = 1.0 if i < 25 else 0.1
            replay.add((torch.randn(10), torch.tensor(i)), priority=priority)

        # Sample should favor high priority
        batch = replay.sample(20)
        assert len(batch) == 20

    def test_ewc_initialization(self):
        """Test EWC initialization."""
        model = SimpleModel()

        ewc = ElasticWeightConsolidation(model, lambda_ewc=100.0)

        assert ewc.lambda_ewc == 100.0
        assert len(ewc.fisher_info) == 0
        assert len(ewc.optimal_params) == 0

    def test_ewc_loss(self):
        """Test EWC loss computation."""
        model = SimpleModel()
        ewc = ElasticWeightConsolidation(model, lambda_ewc=100.0)

        # Initialize Fisher information (dummy)
        for name, param in model.named_parameters():
            ewc.fisher_info[name] = torch.ones_like(param) * 0.1
            ewc.optimal_params[name] = param.data.clone()

        # Compute loss
        loss = ewc.ewc_loss()

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_online_meta_learner(self):
        """Test online meta-learning."""
        model = SimpleModel()

        learner = OnlineMetaLearner(
            model,
            meta_lr=1e-3,
            use_ewc=False  # Disable EWC for faster testing
        )

        # Create new data
        new_data = [(torch.randn(10), torch.randn(10)) for _ in range(10)]

        # Adapt
        stats = learner.online_adapt(new_data, task_boundary=False)

        assert "adaptation_count" in stats
        assert stats["adaptation_count"] == 1
        assert "replay_buffer_size" in stats

    def test_continual_learning_trainer(self):
        """Test continual learning trainer."""
        model = SimpleModel()

        trainer = ContinualLearningTrainer(
            model,
            strategy="replay",
            meta_lr=1e-3
        )

        assert trainer.task_id == 0
        assert trainer.strategy == "replay"


@pytest.mark.advanced
@pytest.mark.slow
class TestIntegration:
    """Integration tests for advanced features."""

    def test_moead_with_neural_symbolic(self):
        """Test MOEA/D with neural-symbolic models."""
        # This would test combining multi-objective optimization
        # with neural-symbolic models
        pass

    def test_compressed_continual_learning(self):
        """Test compressed models with continual learning."""
        model = SimpleModel()

        # Compress
        pipeline = CompressionPipeline(model)
        compressed_model, _ = pipeline.compress(
            pruning_ratio=0.3,
            quantization_method="dynamic"
        )

        # Continual learning with compressed model
        # (Note: QAT would be needed for proper continual learning with quantized models)
        assert compressed_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "advanced"])
