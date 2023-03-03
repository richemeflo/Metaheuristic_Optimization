import numpy as np
import time

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, pop_size=50, mutation_factor=0.8, crossover_probability=0.7, max_iter=1000, tol=1e-6):
        """
        Differential Evolution optimizer constructor.

        Parameters
        ----------
        objective_function : function
            The function to be optimized.
        bounds : tuple
            The bounds of the input variables. It should be a tuple of tuples, where each inner tuple is
            a pair of minimum and maximum bounds for a variable.
        pop_size : int, optional
            The population size. Default is 50.
        mutation_factor : float, optional
            The mutation factor. Default is 0.8.
        crossover_probability : float, optional
            The crossover probability. Default is 0.7.
        max_iter : int, optional
            The maximum number of iterations. Default is 1000.
        tol : float, optional
            The tolerance level for convergence. Default is 1e-6.
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability
        self.max_iter = max_iter
        self.tol = tol
        self.num_variables = len(bounds)

    def optimize(self, verbose=False):
        """
        Runs the differential evolution algorithm and returns the best solution found.
        """
        if verbose:
            print('Running the Differential Evolution algorithm...')

        # Start the timer
        start = time.time()

        # Initialize the population
        pop = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.pop_size, self.num_variables))

        # Evaluate the objective function for each individual
        fitness = np.array([self.objective_function(ind) for ind in pop])

        # Find the best individual in the initial population
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]

        # Start the main loop
        for i in range(self.max_iter):
            # Initialize an empty array to hold the new population
            new_pop = np.empty_like(pop)

            # Loop over each individual in the population
            for j, x in enumerate(pop):
                # Select three random individuals from the population, excluding the current one
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, size=3, replace=False)]

                # Mutate the current individual
                y = a + self.mutation_factor * (b - c)

                # Apply crossover to create the trial individual
                mask = np.random.rand(self.num_variables) < self.crossover_probability
                trial = np.where(mask, y, x)

                # Ensure the trial individual is within the bounds of the search space
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

                # Evaluate the fitness of the trial individual
                trial_fitness = self.objective_function(trial)

                # If the trial individual is better than the current one, replace it in the population
                if trial_fitness < fitness[j]:
                    new_pop[j] = trial
                    fitness[j] = trial_fitness
                else:
                    new_pop[j] = x

            # Replace the old population with the new one
            pop = new_pop

            # Update the best individual if necessary
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best = pop[best_idx]
                best_fitness = fitness[best_idx]
        
        # Stop the timer
        end = time.time()
        if verbose:
            print("The algorithm has finished running.")

        # Calculate the total time taken to run the algorithm
        timer = end - start

        # Print the results
        if verbose:
            print('Best solution:', best,' and the value is ',best_fitness)
            print('Time taken to run the algorithm is ',timer,' ms.')
            
        # Return the best solution found
        return best, best_fitness, timer
    

#create test function
if __name__ == '__main__':
    # Import the necessary packages for the example
    from scipy.optimize import rosen

    # Define the bounds of the search space
    bounds = [[-5, 5], [-5, 5]]

    # Convert the bounds to a numpy array
    bounds_array = np.array(bounds)

    # Create an instance of DifferentialEvolution
    de = DifferentialEvolution(rosen, bounds_array,max_iter=1000)

    # Run the optimization for 1000 iterations
    res = de.optimize(True)
