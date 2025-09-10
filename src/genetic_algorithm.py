import numpy as np
import random
import multiprocessing as mp
from functools import partial

class GeneticAlgorithm:
    def __init__(self, fitness_function, bounds, population_size=50, 
                 mutation_rate=0.1, mutation_scale=0.02, elite_size=2, crossover_rate=0.8):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.parameters = list(bounds.keys())
        self.fitness_history = []
        
    def create_individual(self):
        return {param: np.random.uniform(low, high) for param, (low, high) in self.bounds.items()}
    
    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_fitness(self, individual):
        try:
            fitness = self.fitness_function(individual)
            return -np.inf if np.isnan(fitness) else fitness
        except Exception as e:
            return -np.inf
    
    def rank_population(self, population, processes=1):
        if processes > 1:
            with mp.Pool(processes=processes) as pool:
                fitness_results = list(pool.map(self.fitness_function, population))
        else:
            fitness_results = [self.fitness_function(ind) for ind in population]
        return list(enumerate(fitness_results))
    
    def selection(self, pop_ranked):
        selection_results = []
        
        for i in range(self.elite_size):
            selection_results.append(pop_ranked[i][0])
        
        for _ in range(len(pop_ranked) - self.elite_size):
            selection_results.append(self.tournament_selection(pop_ranked))
        
        return selection_results
    
    def tournament_selection(self, pop_ranked, tournament_size=5):
        tournament_indices = random.sample(range(len(pop_ranked)), tournament_size)
        tournament = [(i, pop_ranked[i][1]) for i in tournament_indices]
        return max(tournament, key=lambda x: x[1])[0]
    
    def crossover(self, parent1, parent2):
        child = {}
        alpha = 0.5
        
        for param in self.parameters:
            if np.random.random() < self.crossover_rate:
                min_val = min(parent1[param], parent2[param])
                max_val = max(parent1[param], parent2[param])
                range_val = max_val - min_val
                
                child[param] = np.random.uniform(
                    min_val - alpha * range_val,
                    max_val + alpha * range_val
                )
                
                low, high = self.bounds[param]
                child[param] = np.clip(child[param], low, high)
            else:
                child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
        
        return child
    
    def mutate(self, individual):
        mutated_individual = individual.copy()
        for param in self.parameters:
            if np.random.random() < self.mutation_rate:
                low, high = self.bounds[param]
                # Используем mutation_scale для контроля размера изменений
                std = (high - low) * self.mutation_scale
                mutated_individual[param] += np.random.normal(0, std)
                mutated_individual[param] = np.clip(mutated_individual[param], low, high)
        return mutated_individual
    
    def create_children(self, selected_population, population):
        children = []
        
        for i in range(self.elite_size):
            children.append(population[selected_population[i]])
        
        for i in range(self.elite_size, len(selected_population)):
            parent1_idx = selected_population[i]
            parent2_idx = selected_population[np.random.randint(0, len(selected_population))]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            child = self.crossover(parent1, parent2)
            children.append(child)
        
        return children
    
    def evolve(self, population, processes=1):
        pop_ranked = self.rank_population(population, processes)
        pop_ranked.sort(key=lambda x: x[1], reverse=True)
        selected = self.selection(pop_ranked)
        children = self.create_children(selected, population)
        next_generation = [self.mutate(child) for child in children]
        return next_generation, pop_ranked
    
    def run(self, generations=100, processes=1):
        population = self.create_population()
        best_fitness = -np.inf
        best_individual = None
        
        for i in range(generations):
            population, pop_ranked = self.evolve(population, processes)
            current_best_fitness = pop_ranked[0][1]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[pop_ranked[0][0]]
            
            self.fitness_history.append(current_best_fitness)
            print(f"Generation {i+1}/{generations}: Best Fitness = {current_best_fitness:.4f}")
        
        return population, best_individual, best_fitness