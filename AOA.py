import numpy as np
import time

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def AOA(Positions, fobj, lb, ub, max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    # Initialize the population
    epsilon = 1e-6
    gamma = 0.1
    #dim = np.size(lb)
    pop = np.random.uniform(lb, ub, (N, dim))

    Convergence_curve = np.zeros((max_iter, 1))

    # Evaluate the objective function
    pop_val = fobj(pop)

    # Initialize the best solution
    best_idx = np.argmin(pop_val)
    best_sol = pop[best_idx]
    best_val = pop_val[best_idx]

    t = 0
    ct = time.time()
    while t < max_iter:
    # Main loop
        for t in range(max_iter):

            # Compute the buoyancy force
            pop_diff = np.expand_dims(pop - best_sol, axis=1)
            pop_dist = np.sqrt(np.sum(pop_diff ** 2, axis=2))
            pop_weight = np.exp(-gamma * pop_dist)
            buoy_force = np.sum(pop_weight[:, :, np.newaxis] * pop_diff, axis=0)

            # Compute the random diffusion
            diff = np.random.randn(N, dim)

            # Compute the resultant force
            res_force = buoy_force + diff

            # Update the position and evaluate the objective function
            pop_new = pop + res_force
            pop_new = np.clip(pop_new, lb, ub)
            pop_new_val = fobj(pop)

            # Update the best solution
            best_idx = np.argmin(pop_new_val)
            if pop_new_val[best_idx] < best_val:
                best_sol = pop_new[best_idx]
                best_val = pop_new_val[best_idx]

            # Check for convergence
            if np.linalg.norm(res_force) < epsilon:
                break

            # Update the population
            pop = pop_new
            pop_val = pop_new_val
        Convergence_curve[t] = best_val
        t = t + 1

    best_val = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_sol, Convergence_curve, best_val, ct
