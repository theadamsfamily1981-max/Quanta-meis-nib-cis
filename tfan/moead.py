"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) for TFAN.
Handles 50+ objectives with Pareto-optimal solution generation.

Based on: Zhang & Li "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition" (2007)
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from .io import write_json


@dataclass
class Objective:
    """Single objective specification."""
    name: str
    minimize: bool = True  # True for minimization, False for maximization
    weight: float = 1.0
    constraint_fn: Optional[Callable] = None  # Optional constraint function


class WeightVector:
    """Weight vector for objective decomposition."""

    @staticmethod
    def generate_uniform(num_objectives: int, num_vectors: int) -> np.ndarray:
        """
        Generate uniformly distributed weight vectors using Das-Dennis method.

        Args:
            num_objectives: Number of objectives
            num_vectors: Number of weight vectors to generate

        Returns:
            Array of weight vectors [num_vectors, num_objectives]
        """
        # Simplex-lattice design
        H = num_vectors // num_objectives

        weights = []

        def generate_recursive(m, s, w):
            """Recursively generate weight vectors."""
            if m == 1:
                w.append(s)
                weights.append(w.copy())
            else:
                for i in range(s + 1):
                    w_new = w.copy()
                    w_new.append(i)
                    generate_recursive(m - 1, s - i, w_new)

        generate_recursive(num_objectives, H, [])

        # Normalize
        weights = np.array(weights, dtype=np.float32)
        weights = weights / H

        # If we don't have enough, add random vectors
        if len(weights) < num_vectors:
            remaining = num_vectors - len(weights)
            random_weights = np.random.dirichlet(np.ones(num_objectives), remaining)
            weights = np.vstack([weights, random_weights])

        return weights[:num_vectors]


class MOEADOptimizer:
    """
    MOEA/D optimizer for multi-objective optimization.

    Decomposes multi-objective problem into scalar subproblems using weight vectors.
    """

    def __init__(self,
                 objectives: List[Objective],
                 population_size: int = 100,
                 num_neighbors: int = 20,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 decomposition: str = "tchebycheff"):
        """
        Args:
            objectives: List of objectives to optimize
            population_size: Size of population
            num_neighbors: Number of neighboring subproblems
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            decomposition: Decomposition method ('tchebycheff', 'weighted_sum', 'pbi')
        """
        self.objectives = objectives
        self.num_objectives = len(objectives)
        self.population_size = population_size
        self.num_neighbors = num_neighbors
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.decomposition = decomposition

        # Generate weight vectors
        self.weight_vectors = WeightVector.generate_uniform(
            self.num_objectives,
            population_size
        )

        # Initialize neighbor indices for each subproblem
        self.neighbors = self._compute_neighbors()

        # Reference point (ideal point)
        self.reference_point = np.full(self.num_objectives, np.inf)

        # Population storage
        self.population = []
        self.objective_values = []

    def _compute_neighbors(self) -> np.ndarray:
        """
        Compute neighboring subproblems based on weight vector distances.

        Returns:
            Array of neighbor indices [population_size, num_neighbors]
        """
        # Compute pairwise distances between weight vectors
        distances = np.zeros((self.population_size, self.population_size))

        for i in range(self.population_size):
            for j in range(self.population_size):
                distances[i, j] = np.linalg.norm(
                    self.weight_vectors[i] - self.weight_vectors[j]
                )

        # Get k nearest neighbors for each weight vector
        neighbors = np.argsort(distances, axis=1)[:, :self.num_neighbors]

        return neighbors

    def _tchebycheff_decomposition(self,
                                   objective_vals: np.ndarray,
                                   weight: np.ndarray) -> float:
        """
        Tchebycheff decomposition function.

        Args:
            objective_vals: Objective values
            weight: Weight vector

        Returns:
            Scalar fitness value
        """
        # Ensure reference point is updated
        weighted_diff = weight * np.abs(objective_vals - self.reference_point)
        return np.max(weighted_diff)

    def _weighted_sum_decomposition(self,
                                    objective_vals: np.ndarray,
                                    weight: np.ndarray) -> float:
        """Weighted sum decomposition."""
        return np.sum(weight * objective_vals)

    def _pbi_decomposition(self,
                          objective_vals: np.ndarray,
                          weight: np.ndarray,
                          theta: float = 5.0) -> float:
        """
        Penalty-based boundary intersection (PBI) decomposition.

        Args:
            objective_vals: Objective values
            weight: Weight vector
            theta: Penalty parameter

        Returns:
            Scalar fitness value
        """
        # Normalize weight
        norm_weight = weight / (np.linalg.norm(weight) + 1e-10)

        # Distance along direction
        diff = objective_vals - self.reference_point
        d1 = np.abs(np.dot(diff, norm_weight))

        # Distance to direction
        d2 = np.linalg.norm(diff - d1 * norm_weight)

        return d1 + theta * d2

    def decompose(self,
                  objective_vals: np.ndarray,
                  weight_idx: int) -> float:
        """
        Decompose multi-objective into scalar using specified method.

        Args:
            objective_vals: Array of objective values
            weight_idx: Index of weight vector to use

        Returns:
            Scalar fitness value
        """
        weight = self.weight_vectors[weight_idx]

        if self.decomposition == "tchebycheff":
            return self._tchebycheff_decomposition(objective_vals, weight)
        elif self.decomposition == "weighted_sum":
            return self._weighted_sum_decomposition(objective_vals, weight)
        elif self.decomposition == "pbi":
            return self._pbi_decomposition(objective_vals, weight)
        else:
            raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def update_reference_point(self, objective_vals: np.ndarray):
        """Update reference point (ideal point) if better values found."""
        self.reference_point = np.minimum(self.reference_point, objective_vals)

    def evaluate_population(self,
                           eval_fn: Callable,
                           population: List) -> np.ndarray:
        """
        Evaluate population on all objectives.

        Args:
            eval_fn: Function that takes individual and returns objective values
            population: List of individuals

        Returns:
            Array of objective values [population_size, num_objectives]
        """
        objective_vals = np.zeros((len(population), self.num_objectives))

        for i, individual in enumerate(population):
            vals = eval_fn(individual)
            objective_vals[i] = vals

            # Update reference point
            self.update_reference_point(vals)

        return objective_vals

    def get_pareto_front(self) -> Tuple[List, np.ndarray]:
        """
        Extract Pareto-optimal solutions from current population.

        Returns:
            (pareto_individuals, pareto_objective_values)
        """
        if len(self.population) == 0:
            return [], np.array([])

        objective_vals = np.array(self.objective_values)

        # Find non-dominated solutions
        is_pareto = np.ones(len(self.population), dtype=bool)

        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i == j:
                    continue

                # Check if j dominates i
                if np.all(objective_vals[j] <= objective_vals[i]) and \
                   np.any(objective_vals[j] < objective_vals[i]):
                    is_pareto[i] = False
                    break

        pareto_indices = np.where(is_pareto)[0]
        pareto_individuals = [self.population[i] for i in pareto_indices]
        pareto_vals = objective_vals[pareto_indices]

        return pareto_individuals, pareto_vals

    def compute_hypervolume(self, reference: np.ndarray) -> float:
        """
        Compute hypervolume indicator for Pareto front.

        Args:
            reference: Reference point for hypervolume

        Returns:
            Hypervolume value
        """
        _, pareto_vals = self.get_pareto_front()

        if len(pareto_vals) == 0:
            return 0.0

        # Simple 2D hypervolume calculation
        # For higher dimensions, use specialized algorithms
        if self.num_objectives == 2:
            # Sort by first objective
            sorted_idx = np.argsort(pareto_vals[:, 0])
            sorted_vals = pareto_vals[sorted_idx]

            hv = 0.0
            prev_x = reference[0]

            for val in sorted_vals:
                hv += (prev_x - val[0]) * (reference[1] - val[1])
                prev_x = val[0]

            return hv
        else:
            # For >2 objectives, use approximate method
            # This is a simplified version
            volumes = []
            for val in pareto_vals:
                volume = np.prod(reference - val)
                volumes.append(volume)

            return sum(volumes) / len(volumes)


class MOEADTrainer:
    """
    MOEA/D-based multi-objective trainer for TFAN.
    Integrates with existing TFAN training infrastructure.
    """

    def __init__(self,
                 model_factory: Callable,
                 objectives: List[Objective],
                 population_size: int = 100,
                 num_neighbors: int = 20,
                 max_generations: int = 100,
                 log_dir: str = "logs/moead"):
        """
        Args:
            model_factory: Function that creates a new model instance
            objectives: List of objectives to optimize
            population_size: Population size
            num_neighbors: Number of neighbors
            max_generations: Maximum number of generations
            log_dir: Directory for logs
        """
        self.model_factory = model_factory
        self.objectives = objectives
        self.population_size = population_size
        self.max_generations = max_generations
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create MOEA/D optimizer
        self.optimizer = MOEADOptimizer(
            objectives=objectives,
            population_size=population_size,
            num_neighbors=num_neighbors
        )

        # Statistics
        self.generation = 0
        self.hypervolume_history = []
        self.pareto_history = []

    def evolve(self,
               eval_fn: Callable,
               initial_population: Optional[List] = None) -> Dict:
        """
        Run MOEA/D evolution.

        Args:
            eval_fn: Function to evaluate individuals on objectives
            initial_population: Optional initial population

        Returns:
            Evolution statistics
        """
        # Initialize population
        if initial_population is None:
            population = [self.model_factory() for _ in range(self.population_size)]
        else:
            population = initial_population

        self.optimizer.population = population

        # Evaluate initial population
        objective_vals = self.optimizer.evaluate_population(eval_fn, population)
        self.optimizer.objective_values = objective_vals.tolist()

        print(f"=== MOEA/D Evolution ({self.num_objectives} objectives) ===")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}\n")

        # Evolution loop
        for gen in range(self.max_generations):
            self.generation = gen

            # Evolve each subproblem
            for i in range(self.population_size):
                # Select neighbors
                neighbors = self.optimizer.neighbors[i]

                # Reproduction (simplified - using crossover)
                parent_idx = np.random.choice(neighbors, size=2, replace=False)

                # Create offspring (this is problem-specific)
                # For neural networks, could use parameter averaging
                offspring = self._reproduce(
                    population[parent_idx[0]],
                    population[parent_idx[1]]
                )

                # Evaluate offspring
                offspring_vals = eval_fn(offspring)
                self.optimizer.update_reference_point(offspring_vals)

                # Update neighbors
                for neighbor_idx in neighbors:
                    # Decompose objectives
                    current_fitness = self.optimizer.decompose(
                        np.array(self.optimizer.objective_values[neighbor_idx]),
                        neighbor_idx
                    )

                    offspring_fitness = self.optimizer.decompose(
                        offspring_vals,
                        neighbor_idx
                    )

                    # Replace if better
                    if offspring_fitness < current_fitness:
                        population[neighbor_idx] = offspring
                        self.optimizer.objective_values[neighbor_idx] = offspring_vals.tolist()

            # Track Pareto front
            pareto_individuals, pareto_vals = self.optimizer.get_pareto_front()
            self.pareto_history.append(len(pareto_individuals))

            # Compute hypervolume
            reference = np.ones(len(self.objectives)) * 10.0  # Problem-specific
            hv = self.optimizer.compute_hypervolume(reference)
            self.hypervolume_history.append(hv)

            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{self.max_generations} | "
                      f"Pareto size: {len(pareto_individuals)} | "
                      f"Hypervolume: {hv:.4f}")

        # Final results
        pareto_individuals, pareto_vals = self.optimizer.get_pareto_front()

        print(f"\n=== Evolution Complete ===")
        print(f"Final Pareto front size: {len(pareto_individuals)}")
        print(f"Final hypervolume: {self.hypervolume_history[-1]:.4f}")

        # Save results
        results = {
            "num_objectives": len(self.objectives),
            "population_size": self.population_size,
            "generations": self.max_generations,
            "final_pareto_size": len(pareto_individuals),
            "hypervolume_history": self.hypervolume_history,
            "pareto_size_history": self.pareto_history,
            "pareto_objective_values": pareto_vals.tolist()
        }

        results_path = self.log_dir / "moead_results.json"
        write_json(results_path, results)

        return results

    def _reproduce(self, parent1, parent2):
        """
        Reproduce two parent models to create offspring.
        Uses parameter averaging for neural networks.
        """
        offspring = self.model_factory()

        # Average parameters (simple crossover)
        with torch.no_grad():
            for param_off, param1, param2 in zip(
                offspring.parameters(),
                parent1.parameters(),
                parent2.parameters()
            ):
                # Uniform crossover
                mask = torch.rand_like(param_off) < 0.5
                param_off.copy_(torch.where(mask, param1, param2))

                # Mutation
                if np.random.rand() < self.optimizer.mutation_prob:
                    noise = torch.randn_like(param_off) * 0.01
                    param_off.add_(noise)

        return offspring


if __name__ == "__main__":
    # Demo MOEA/D
    print("=== MOEA/D Demo ===\n")

    # Define objectives
    objectives = [
        Objective(name="loss", minimize=True),
        Objective(name="latency", minimize=True),
        Objective(name="complexity", minimize=True),
        Objective(name="accuracy", minimize=False),  # Maximize accuracy
    ]

    # Test weight vector generation
    weights = WeightVector.generate_uniform(num_objectives=4, num_vectors=50)
    print(f"Generated {len(weights)} weight vectors for {len(objectives)} objectives")
    print(f"Weight vector shape: {weights.shape}")
    print(f"Example weights:\n{weights[:5]}\n")

    # Test MOEA/D optimizer
    optimizer = MOEADOptimizer(
        objectives=objectives,
        population_size=50,
        num_neighbors=10
    )

    print(f"MOEA/D initialized with {optimizer.population_size} subproblems")
    print(f"Decomposition method: {optimizer.decomposition}")
