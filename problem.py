import numpy as np

class MKProblem:

    def __init__(self, N : int, m : int, alpha : float):
        self.N, self.m, self.alpha = N, m, alpha

    def benchmark_mkp(idx : int = 0):
        raise NotImplementedError

    def random_mkp(self) -> tuple:                      
        """Generates the expansions in m dimensions of each of N objects.
        
        Parameters
        -----
        None
        """
        self.weights = np.random.uniform(0, 1000, (self.m, self.N))
        self.constraints = self.alpha * np.sum(self.w, axis = 1)
        self.values = np.random.rand(self.N)
  
    def fitness(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the fitness of any assignment of items.

       Parameters
        -----
        x: np.ndarray
        
        Returns
        -----
        The fitnesses of each solution.
        """

        # --- inherently includes the death penalty ---
        return np.where(self.test_constraint(x), x@self.values, 0)
    
    def test_constraint(self, x: np.ndarray) -> np.ndarray:
        """
        Determines whether the given solutions violates any constraints on the problem.

        Args:
            x: torch.Tensor
                The solutions which will be tested.

        Returns
        -----
        vector of boolean (0,1) indicating the feasibilties of the solutions
        """
        return (x@self.weights.T < self.constraints[None,:]).all(dim=1)
    
    