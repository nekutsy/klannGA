import matplotlib.pyplot as plt
import numpy as np
import argparse
from simulation import simulate_single_system, create_system
from genetic_algorithm import GeneticAlgorithm
from visualization import plot_multiple_trajectories, animate_system
import multiprocessing as mp
from functools import partial

def fitness_function_wrapper(individual):
    try:
        analysis, trajectory = simulate_single_system(
            x=individual['x'],
            y1=individual['y1'],
            y2=individual['y2'],
            rod_lengths=[individual[f'rod{i}_length'] for i in range(2, 8)],
            rigid_angles=[individual['rigid1_angle'], individual['rigid2_angle']]
        )
        
        if (analysis['straight_length'] < 20 or 
            not analysis.get('is_closed', False) or
            np.isnan(analysis['straight_length']) or
            np.isnan(analysis['straightness']) or
            np.isnan(analysis['slope'])):
            return 0.0
            
        fitness = (analysis['straight_length'] * analysis['straightness'] ** 8)
        fitness /= (1 + abs(analysis["slope"])) ** 8
        
        return max(0.0, fitness)
    except Exception as e:
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize rod system parameters')
    parser.add_argument('--population', type=int, default=60, help='Population size for GA')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations for GA')
    parser.add_argument('--mutation', type=float, default=0.1, help='Mutation rate for GA')
    parser.add_argument('--mutation-scale', type=float, default=0.02, help='Scale of mutations (smaller = smaller changes)')
    parser.add_argument('--elite', type=int, default=5, help='Elite size for GA')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(), help='Number of processes to use')
    parser.add_argument('--animate', action='store_true', help='Show animation of the best system')
    args = parser.parse_args()
    
    initial_x = -37.5
    initial_y1 = 17.5
    initial_y2 = -17.5
    
    initial_rod2_length = 24.8
    initial_rod3_length = 17.7
    initial_rod4_length = 39.2
    initial_rod5_length = 30.3
    initial_rod6_length = 36.1
    initial_rod7_length = 66.8
    
    initial_rigid1_angle = np.pi - np.pi/24
    initial_rigid2_angle = 195 * np.pi/180
    
    bounds = {
        'x': (-40, -35),
        'y1': (15, 20),
        'y2': (-20, -15),
        'rod2_length': (20, 30),
        'rod3_length': (15, 25),
        'rod4_length': (35, 45),
        'rod5_length': (25, 35),
        'rod6_length': (30, 40),
        'rod7_length': (60, 70),
        'rigid1_angle': (150 * np.pi/180, 210 * np.pi/180),
        'rigid2_angle': (150 * np.pi/180, 210 * np.pi/180)
    }
    
    initial_individual = {
        'x': initial_x,
        'y1': initial_y1,
        'y2': initial_y2,
        'rod2_length': initial_rod2_length,
        'rod3_length': initial_rod3_length,
        'rod4_length': initial_rod4_length,
        'rod5_length': initial_rod5_length,
        'rod6_length': initial_rod6_length,
        'rod7_length': initial_rod7_length,
        'rigid1_angle': initial_rigid1_angle,
        'rigid2_angle': initial_rigid2_angle
    }

    class CustomGeneticAlgorithm(GeneticAlgorithm):
        def __init__(self, initial_individual, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_individual = initial_individual

        def create_population(self):
            population = [self.initial_individual]
            for _ in range(self.population_size - 1):
                population.append(self.create_individual())
            return population

        def evolve(self, population, processes=1):
            next_generation, pop_ranked = super().evolve(population, processes)
            next_generation[0] = self.initial_individual
            return next_generation, pop_ranked
    
    fitness_func = fitness_function_wrapper
    
    ga = CustomGeneticAlgorithm(
        initial_individual=initial_individual,
        fitness_function=fitness_func,
        bounds=bounds,
        population_size=args.population,
        mutation_rate=args.mutation,
        mutation_scale=args.mutation_scale,
        elite_size=args.elite
    )
    
    final_population, best_individual, best_fitness = ga.run(
        generations=args.generations,
        processes=args.processes
    )
    
    ranked_population = ga.rank_population(final_population, processes=args.processes)
    top_indices = [idx for idx, _ in ranked_population[:5]]
    top_individuals = [final_population[i] for i in top_indices]
    
    trajectories = []
    labels = []
    analyses = []
    
    for i, individual in enumerate(top_individuals):
        rod_lengths = [
            individual['rod2_length'],
            individual['rod3_length'],
            individual['rod4_length'],
            individual['rod5_length'],
            individual['rod6_length'],
            individual['rod7_length']
        ]
        
        rigid_angles = [
            individual['rigid1_angle'],
            individual['rigid2_angle']
        ]
        
        analysis, trajectory = simulate_single_system(
            x=individual['x'],
            y1=individual['y1'],
            y2=individual['y2'],
            rod_lengths=rod_lengths,
            rigid_angles=rigid_angles
        )
        
        if (np.isnan(analysis['straight_length']) or 
            np.isnan(analysis['straightness']) or
            np.isnan(analysis['slope'])):
            continue
            
        trajectories.append(trajectory)
        analyses.append(analysis)
        fitness = fitness_function_wrapper(individual)
        labels.append(f"Top {i+1}: Fitness={fitness:.2f}, Length={analysis['straight_length']:.2f}")
    
    if trajectories:
        plot_multiple_trajectories(trajectories, labels=labels)
        
        best_individual = top_individuals[0]
        print("Optimized parameters:")
        print(f"x: {best_individual['x']:.2f}")
        print(f"y1: {best_individual['y1']:.2f}")
        print(f"y2: {best_individual['y2']:.2f}")
        print(f"rod2_length: {best_individual['rod2_length']:.2f}")
        print(f"rod3_length: {best_individual['rod3_length']:.2f}")
        print(f"rod4_length: {best_individual['rod4_length']:.2f}")
        print(f"rod5_length: {best_individual['rod5_length']:.2f}")
        print(f"rod6_length: {best_individual['rod6_length']:.2f}")
        print(f"rod7_length: {best_individual['rod7_length']:.2f}")
        print(f"rigid1_angle: {best_individual['rigid1_angle']:.4f} rad ({np.degrees(best_individual['rigid1_angle']):.2f}°)")
        print(f"rigid2_angle: {best_individual['rigid2_angle']:.4f} rad ({np.degrees(best_individual['rigid2_angle']):.2f}°)")
        
        print("\nPerformance metrics for best individual:")
        for key, value in analyses[0].items():
            print(f"{key}: {value:.4f}")
            
        if args.animate or True:
            best_rod_lengths = [
                best_individual['rod2_length'],
                best_individual['rod3_length'],
                best_individual['rod4_length'],
                best_individual['rod5_length'],
                best_individual['rod6_length'],
                best_individual['rod7_length']
            ]
            
            best_rigid_angles = [
                best_individual['rigid1_angle'],
                best_individual['rigid2_angle']
            ]
            
            system = create_system(
                x=best_individual['x'],
                y1=best_individual['y1'],
                y2=best_individual['y2'],
                rod_lengths=best_rod_lengths,
                rigid_angles=best_rigid_angles
            )
            
            _, trajectory = simulate_single_system(
                x=best_individual['x'],
                y1=best_individual['y1'],
                y2=best_individual['y2'],
                rod_lengths=best_rod_lengths,
                rigid_angles=best_rigid_angles
            )
            
            animate_system(system, trajectory)
    else:
        print("No valid trajectories found in top individuals")
    
    plt.figure(figsize=(10, 6))
    plt.plot(ga.fitness_history)
    plt.title("Fitness History")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show()