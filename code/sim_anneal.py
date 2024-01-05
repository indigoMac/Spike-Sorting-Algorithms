import numpy as np
from random import random, randint
from sklearn.neighbors import KNeighborsClassifier
from KNN_main import Spike_object

class SimulatedAnnealing:
    """
    Optimizes the K-Nearest Neighbors classifier parameters using simulated annealing.
    """

    def __init__(self):
        # Initialise variables for SA
        self.alpha = 0.1
        self.iters = 100
        self.d = 0.25

    def optimize(self):
        """
        Main optimization function that finds the best KNN parameters.
        """
        # Create a random initial solution within valid range
        initial_solution = [randint(1, 20), randint(1, 3)]
        solution = self.anneal(initial_solution)

        # Optimized parameters
        k = int(round(solution[0]))
        p = int(round(solution[1]))

        return k, p

    def anneal(self, solution):
        """
        Runs the main simulated annealing loop to optimize parameters.
        """
        old_cost = self.cost(solution)
        t = 1.0
        t_min = 0.001

        while t > t_min:
            for _ in range(self.iters):
                new_solution = self.neighbor(solution)
                new_cost = self.cost(new_solution)
                ap = self.acceptance_probability(old_cost, new_cost, t)
                if ap > random():
                    solution = new_solution
                    old_cost = new_cost

            t *= self.alpha

        return solution

    def acceptance_probability(self, old_cost, new_cost, t):
        """
        Calculate the acceptance probability for a new solution.
        """
        return np.exp((old_cost - new_cost) / t)

    def cost(self, solution):
        """
        Calculate the cost of a solution based on the KNN classifier's performance.
        """
        k, p = int(round(solution[0])), int(round(solution[1]))
        spike_obj = Spike_object()
        spike_obj.load_data('D1.mat', train=True)

        # Update the classifier with the new parameters
        spike_obj.classifier = KNeighborsClassifier(n_neighbors=k, p=p)

        # Prepare the data for KNN classifier
        windows = spike_obj.create_window()

        # Split data and train the classifier for cross-validation
        mean_score, _ = spike_obj.cross_validate()

        return 1 - mean_score  # cost as inverse of accuracy

    def neighbor(self, solution):
        """
        Generate a neighboring solution.
        """
        delta = np.random.random(2)
        scale = np.full(2, 2 * self.d)
        offset = np.full(2, 1.0 - self.d)

        m = delta * scale + offset
        solution[0] *= m[0]
        solution[1] *= m[1]

        return solution

if __name__ == '__main__':
    sa = SimulatedAnnealing()
    optimal_k, optimal_p = sa.optimize()
    print(f"Optimal k: {optimal_k}, Optimal p: {optimal_p}")
