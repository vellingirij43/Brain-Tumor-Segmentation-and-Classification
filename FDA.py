import time
import numpy as np


# Flow Direction Algorithm (FDA)

def FDA(alpha, fobj, lb, ub, maxiter):
    N, dim = alpha.shape[0], alpha.shape[1]
    Best_fitness = float('inf')
    # Initialize the Positions of flows
    beta = 1  # Nighbourhood points
    flow_x = N  # initialization(alpha, dim, ub, lb)
    neighbor_x = np.zeros((beta, dim))
    newflow_x = np.zeros((flow_x))
    newfitness_flow = np.zeros((flow_x))
    ConvergenceCurve = np.zeros((1, maxiter))
    fitness_flow = np.zeros((dim, 1))
    fitness_neighbor = np.ones((beta, 1))
    eps = 2.2204e-16
    # calculate fitness function of each flow
    for i in range(len(alpha)):
        fitness_flow[i, :] = fobj(alpha[i, :])  # fitness of each flow end
    BestX = flow_x
    # Initialize velocity of flows
    Vmax = 0.1 * (ub - lb)
    Vmin = -0.1 * (ub - lb)
    t = 0
    ct = time.time()
    # Main loop
    while t < maxiter:
        W = (((1 - 1 * t / maxiter + eps) + (2 * np.random.rand())) * (np.random.randint(1, dim) * t / maxiter) * np.random.randint(1, dim))
        # Update the Position of eachflow
        for i in range(len(alpha)):
            # Produced the Position of neighborhoods around each flow
            for j in range(N):
                Xrand = lb + np.random.randint(1, dim) * (ub - lb)
                delta = W * (np.random.random() - Xrand * flow_x) * np.linalg.norm(BestX - flow_x)
                neighbor_x = flow_x + np.random.rand() * delta
                neighbor_x[j, :] = max(neighbor_x[j, :])
                neighbor_x[j, :] = min(neighbor_x[j, :])
                fitness_neighbor[j] = fobj(neighbor_x[j, :])
                # Sort position of neighborhoods
                [indx] = np.sort(fitness_neighbor)
                # Update position, fitness and velocity of current flow if the fitness of best neighborhood is
                # less than of current flow
                if fitness_neighbor < fitness_flow[i]:
                    #  Calculate slope to neighborhood
                    Sf = (fitness_neighbor[indx[1]] - fitness_flow[i]) / np.sqrt(
                        np.norm(neighbor_x[indx[1], :] - flow_x[i, :]))  # calculating
                    # Update velocity of each flow
                    V = np.random.random() * Sf
                    if V < Vmin:
                        V = - Vmin
                    elif V > Vmax:
                        V = - Vmax
                    # Flow moves to best neighborhood
                    newflow_x[i, :] = flow_x[i, :] + V * (neighbor_x[indx[1], :] - flow_x[i, :]) / np.sqrt(
                        np.linalg.norm(neighbor_x[indx[1], :] - flow_x[i, :]))
                else:
                    # Generate integer random number (r)
                    r = np.random.rand()
                    # Flow moves to r th flow if the fitness of r th flow is less
                    # than current flow
                    if fitness_neighbor < fitness_flow[j]:
                        newflow_x[i, :] = flow_x[i, :] + np.random.randint(1, dim) * (flow_x[r, :] - flow_x[i, :])
                    else:
                        newflow_x = flow_x+ np.random.random() * (BestX - flow_x)

                    # Return the flows that go beyond the boundaries of the search space
                    newflow_x = max(newflow_x[i, :])
                    newflow_x = min(newflow_x[i, :])
                    # Calculate fitness function of new flow
                    newfitness_flow[i] = fobj(newflow_x[i, :])
                # Update current flow
                if newfitness_flow[i] < fitness_flow[i]:
                    flow_x[i, :] = newflow_x[i, :]
                fitness_flow[i] = newfitness_flow[i]
                # Update best flow
                if fitness_flow[i] < Best_fitness:
                    BestX = flow_x[i, :]
                Best_fitness = fitness_flow[i]
        # Oppsotion based learning
        D = []
        for i in range(len(alpha)):
            D[i, :] = min(flow_x[i, :]) + max(flow_x[i, :]) - flow_x[i, :]
            Flag4ub = D[i, :] > ub
            Flag4lb = D[i, :] < lb
            D[i, :] = (D[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            op_fitness = fobj(D[i, :])
            if op_fitness < fitness_flow[i]:
                flow_x[i, :] = D[i, :]
                fitness_flow[i] = op_fitness
                ConvergenceCurve[t] = Best_fitness
        score = ConvergenceCurve[maxiter - 1]
        ct = time.time() - ct
        t = t + 1
    return Best_fitness, ConvergenceCurve, BestX, ct
