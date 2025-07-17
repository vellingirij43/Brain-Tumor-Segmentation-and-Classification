import numpy as np
import time


## Gorilla Troops Optimizer (GTO)
def GTO(population, fobj, VRmin, VRmax, max_iterations):
    N, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    best_solution = np.zeros((1, dim))
    best_fitness = float('inf')
    Convergence_curve = np.zeros((max_iterations, 1))

    t = 0
    ct = time.time()
    for t in range(max_iterations):
        # Evaluate fitness of each individual in the population
        fitness_values = fobj(population[:])

        # Find the index of the best individual
        best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[best_index]

        # Update best solution if the current one is better
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[best_index]

        # Update each individual's position based on the best individual
        for i in range(N):
            if i != best_index:
                population[i] = population[i] + np.random.uniform() * (best_solution - population[i])

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iterations - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
