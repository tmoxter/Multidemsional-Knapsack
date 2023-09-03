import numpy as np
from joblib import Parallel, delayed, memory
from functools import wraps
from torch.utils.tensorboard import SummaryWriter

from problem import MKProblem

class RepairOperator:
    def __init__(self, problem : MKProblem) -> None:
        self.problem = problem
        self.utiliy = np.argsort(problem.values/problem.weights.sum(axis=0))
    
    def __call__(self, x : np.ndarray) -> None:
        self._repair(x)
    
    def _repair(self, x : np.ndarray) -> None:
        valid = self.problem.test_constraint(x)
        for i in self.utility:
            x[:,i] = np.where(~valid & (x[:,i] == 1), 0, x[:,i])
            valid[~valid] = self.problem.test_constraint(x[~valid])
            if valid.all():
                break
        for i in self.utility[::-1]:
            was_added = x[:, i] == 0
            x[:, i] = 1
            valid = self.problem.test_constraint(x)
            x[~valid & was_added, i] = 0

def parallel_runs(function : callable, n_runs : int):
    n_jobs = max(n_runs)
    @wraps(function)
    def wrapper(*args, **kwargs):
        return Parallel(n_jobs)(delayed(function)(*args, **kwargs)
                                for _ in range(n_runs))
    return wrapper

class FitnessLogger(SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir)
        self.n_gen = 0
    
    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            fitness = function(*args, **kwargs)
            self.n_gen += 1
            self.add_scalar("Best v. n_gen", max(fitness), self.n_gen)
            self.add_scalar("Mean v. n_gen", np.mean(fitness), self.n_eval)
            return fitness
        return wrapper
