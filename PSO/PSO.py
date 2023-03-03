import numpy as np
import time

class PSO:
    def __init__(self, obj_func, bounds, num_particles=50, num_iterations=100):
        """
        Initializes the PSO object.

        Parameters:
        obj_func (callable): Objective function to be optimized.
        bounds (numpy.ndarray): A two-dimensional array of shape (n, 2), where n is the number of
                                 dimensions in the parameter space. The ith row should contain the
                                 minimum and maximum values for the ith dimension.
        num_particles (int): Number of particles in the swarm (default: 50).
        num_iterations (int): Maximum number of iterations (default: 100).
        
        """
        self.obj_func = obj_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.num_iterations = num_iterations

    def optimize(self, verbose=False):
        """
        Optimizes the objective function using PSO.

        Parameters:
        verbose (bool): Whether to print progress during optimization (default: False).
        Returns:
        best_position (numpy.ndarray): The best position found.
        best_score (float): The best score found.
        timer (float): The time taken to run the algorithm.
        """

        if verbose:
            print('Running the Particle Swarm Optimization algorithm...')

        # Start the timer
        start = time.time()

        # Initialize the swarm
        self.swarm = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.num_particles, len(self.bounds)))
        self.velocities = np.zeros_like(self.swarm)
        self.particle_best = self.swarm.copy()
        self.particle_best_scores = np.full(self.num_particles, np.inf)
        self.global_best = None
        self.global_best_score = np.inf

        # Run the optimization loop
        for i in range(self.num_iterations):
            # Evaluate the objective function for each particle
            particle_scores = np.array([self.obj_func(p) for p in self.swarm])

            # Update the particle best scores and positions
            improved = particle_scores < self.particle_best_scores
            self.particle_best_scores[improved] = particle_scores[improved]
            self.particle_best[improved] = self.swarm[improved]

            # Update the global best score and position
            j = np.argmin(particle_scores)
            if particle_scores[j] < self.global_best_score:
                self.global_best_score = particle_scores[j]
                self.global_best = self.swarm[j]

            # Update the particle velocities and positions
            r1 = np.random.rand(self.num_particles, 1)
            r2 = np.random.rand(self.num_particles, 1)
            self.velocities = 0.5 * self.velocities + \
                              2.0 * r1 * (self.particle_best - self.swarm) + \
                              2.0 * r2 * (self.global_best - self.swarm)
            self.swarm += self.velocities

            # Enforce the bounds on the swarm
            self.swarm = np.maximum(self.swarm, self.bounds[:, 0])
            self.swarm = np.minimum(self.swarm, self.bounds[:, 1])

        # Stop the timer
        end = time.time()
        if verbose:
            print("The algorithm has finished running.")

        # Calculate the time taken
        timer = end - start

        # Print progress
        if verbose:
            print(f"Iteration {i+1}/{self.num_iterations}: Best score = {self.global_best_score} & Best position = {self.global_best}")
            print(f"Time taken: {timer:f} ms.")

        return self.global_best, self.global_best_score, timer
    

#create test function
if __name__ == '__main__':
    # Import the necessary packages for the example
    from scipy.optimize import rosen

    # Define the bounds of the search space
    bounds = [[-5, 5], [-5, 5]]

    # Convert the bounds to a numpy array
    bounds_array = np.array(bounds)

    pso = PSO(rosen, bounds_array, num_particles=50, num_iterations=1000)
    res= pso.optimize(verbose=True)
