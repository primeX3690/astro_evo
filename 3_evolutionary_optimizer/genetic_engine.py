# 3_evolutionary_optimizer/genetic_engine.py
"""Dependency-free real-valued genetic algorithm - no DEAP."""
import numpy as np
from trajectory_encoding import random_gene, clip_gene
from fitness_functions import evaluate_trajectory


class GeneticEngine:
    def __init__(self, mission_spec, population_size=40, generations=40,
                 mutation_rate=0.3, mutation_sigma=(3.0, 0.5), elite_count=2, seed=42):
        self.mission_spec = mission_spec
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = np.array(mutation_sigma)
        self.elite_count = elite_count
        self.rng = np.random.default_rng(seed)
        self.history = []

    def _init_population(self):
        return [random_gene(self.rng) for _ in range(self.pop_size)]

    def _tournament_select(self, pop, fitnesses, k=3):
        idxs = self.rng.integers(0, len(pop), size=k)
        best_idx = idxs[np.argmax(fitnesses[idxs])]
        return pop[best_idx]

    def _crossover(self, parent_a, parent_b):
        alpha = self.rng.uniform(0, 1, size=2)
        return clip_gene(alpha * parent_a + (1 - alpha) * parent_b)

    def _mutate(self, gene):
        if self.rng.uniform() < self.mutation_rate:
            gene = gene + self.rng.normal(0, self.mutation_sigma)
        return clip_gene(gene)

    def run(self):
        population = self._init_population()
        best_gene, best_fitness, best_info = None, -np.inf, None

        for _ in range(self.generations):
            fitnesses = np.array([evaluate_trajectory(g, self.mission_spec)[0] for g in population])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_gene = population[gen_best_idx].copy()
                best_fitness, best_info = evaluate_trajectory(best_gene, self.mission_spec)
            self.history.append(best_fitness)

            elite_idx = np.argsort(fitnesses)[-self.elite_count:]
            new_population = [population[i].copy() for i in elite_idx]
            while len(new_population) < self.pop_size:
                parent_a = self._tournament_select(population, fitnesses)
                parent_b = self._tournament_select(population, fitnesses)
                child = self._mutate(self._crossover(parent_a, parent_b))
                new_population.append(child)
            population = new_population

        return best_gene, best_fitness, best_info