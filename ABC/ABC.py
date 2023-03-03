import numpy as np
import time

class ABC:
    def __init__(self, obj_func, num_variables, bounds, colony_size, num_iter, limit):
        """
        Artificial Bee Colony optimizer constructor.

        Parameters
        ----------
        obj_func : function
            The function to be optimized.
        num_variables : int
            The number of variables in the objective function.
        bounds : tuple
            The bounds of the input variables. It should be a tuple of tuples, where each inner tuple is
            a pair of minimum and maximum bounds for a variable.
        colony_size : int
            The number of solutions in the colony.
        num_iter : int
            The number of iterations.
        limit : int
            The number of iterations without improvement after which a source is abandoned.
        """
        self.obj_func = obj_func
        self.num_variables = num_variables
        self.bounds = bounds
        self.colony_size = colony_size
        self.num_iter = num_iter
        self.limit = limit
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.colony = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.colony_size, self.num_variables))
        self.fitness_values = np.zeros(self.colony_size)

    def optimize(self,verbose=False):
        """
        Runs the artificial bee colony algorithm and returns the best solution found.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the progress of the algorithm. Default is False.

        Returns
        -------
        best_solution : numpy.ndarray
            The best solution found.
        timer : float
            The time taken to run the algorithm.
        """

        if verbose:
            print('Running the ABC algorithm...')

        # Start the timer
        start = time.time()

        # evaluate the objective function for each source
        for i in range(self.num_iter):
            self.send_employed_bees()
            self.send_onlooker_bees()
            self.send_scout_bees()


            # keep track of the best solution
            for j in range(self.colony_size):
                if self.fitness_values[j] < self.best_fitness:
                    self.best_fitness = self.fitness_values[j]
                    self.best_solution = self.colony[j]

        # Stop the timer
        end = time.time()
        if verbose:
            print("The algorithm has finished running.")
        
        # Calculate the total time taken to run the algorithm
        timer = end - start

        if verbose:
            print('Iteration:', i + 1, 'Best Fitness:', self.best_fitness, ' and Best Solution:', self.best_solution)
            print('Time taken to run the algorithm is ',timer,' ms.')

        return self.best_solution, self.best_fitness, timer

    def send_employed_bees(self):
        """
        Sends employed bees to exploit the sources in the colony.
        """
        for i in range(self.colony_size):
            v = self.get_new_source(i)
            fitness_v = self.obj_func(v)

            if fitness_v < self.fitness_values[i]:
                self.colony[i] = v
                self.fitness_values[i] = fitness_v

    def send_onlooker_bees(self):
        """
        Sends onlooker bees to exploit the best sources in the colony.
        """
        fitness_probs = self.fitness_values.max() - self.fitness_values
        if np.all(fitness_probs == 0): # handle case when all fitness values are the same
            probs = np.ones_like(fitness_probs) / self.colony_size
        else:
            probs = fitness_probs / np.sum(fitness_probs)

        for i in range(self.colony_size):
            # select a source from the colony based on the roulette wheel
            j = np.random.choice(self.colony_size, p=probs)

            # generate a new solution from the selected source
            v = self.get_new_source(j)
            fitness_v = self.obj_func(v)

            # update the colony if the new solution is better
            if fitness_v < self.fitness_values[j]:
                self.colony[j] = v
                self.fitness_values[j] = fitness_v

    def send_scout_bees(self):
        """
        Sends scout bees to discover new sources.
        """
        # find the sources that have exceeded the limit
        limit_exceeds = np.where(self.fitness_values >= self.limit)[0]

        # replace the sources that have exceeded the limit with new random solutions
        for i in limit_exceeds:
            self.colony[i] = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=self.num_variables)
            self.fitness_values[i] = self.obj_func(self.colony[i])

    def get_new_source(self, index):
        # select two sources different from the source at the given index
        sources = [i for i in range(self.colony_size) if i != index]
        idx_a, idx_b = np.random.choice(sources, size=2, replace=False)

        # generate a new source from the selected sources
        phi = np.random.uniform(low=-1, high=1, size=self.num_variables)
        v = self.colony[index] + phi * (self.colony[idx_a] - self.colony[idx_b])

        # apply bounds to the new source
        v = np.clip(v, self.bounds[:, 0], self.bounds[:, 1])

        return v
    


#create test function
if __name__ == '__main__':
    # Import the necessary packages for the example
    from scipy.optimize import rosen

    # Define the bounds of the search space
    bounds = [[-5, 5], [-5, 5]]

    # Convert the bounds to a numpy array
    bounds_array = np.array(bounds)

    abc = ABC(obj_func=rosen, num_variables=2, bounds=bounds_array, colony_size=10, num_iter=1000, limit=100)
    res = abc.optimize(verbose=True)
