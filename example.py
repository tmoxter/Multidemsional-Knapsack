import argparse
from datetime import datetime
from atexit import register

from optimizers import SimpleGA, HillClimber
from problem import MKProblem
from utils import *

SEED = 48

@register
def early_termination():
    # Move into objects to save data?
    pass

def main(config):

    if not config.problem_instance:
        problem = MKProblem(100, 30, 0.25)
        problem.random_mkp()
    if config.log:
        wrap = FitnessLogger('runs/')
        problem.fitness = wrap(problem.fitness)

    @parallel_runs(config.n_runs)
    def run():
        if config.algo == 'GA':
            optimizer = SimpleGA(config.shape, problem,
                        max_evals=config.max_evlas, max_gens=config.max_gens,
                        max_time=config.max_time, verbose=config.verbose)
            optimizer.evolve()
            return optimizer.runData
        
        elif config.algo == 'HC':
            optimizer = HillClimber(config.shape, problem,n_bits=2,
                        max_evals=config.max_evlas, max_gens=config.max_gens,
                        max_time=config.max_time, verbose=config.verbose)
            optimizer.climb()
            return optimizer.runData

    # --- run the algorithm ---
    data = run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple optimimzers for the multidimensional knapsack'
        )
    parser.add_argument('c', '--config', default=None, type=str,
        help='path to a config file (json), overwrites rest of parameters if passed')
    parser.add_argument('-p', '--problem_intance', default=None, type=str,
        help='path to the problem instance, defaults to random intialization')
    parser.add_argument('-s','--shape', default=(1000,100), type=tuple,
                        help='Population size (int) x problem instance size (int)')
    parser.add_argument('-a', '--algo', default='GA', type=str,
                        help='Which heuristic to use (GA / HC)')
    parser.add_argument('-e', '--max_eval', default=None, type=int,
                        help='maximum number of evaluations')
    parser.add_argument('-g', '--max_gen', default=None, type=int,
                        help='maximum number of generations')
    parser.add_argument('-t', '--max_time', default=None, type=float,
                        help='maximum run time')
    parser.add_argument('-v', '--verbose', default=True, type=bool,
                        help='print information')
    parser.add_argument('-r', '--n_runs', default='1', type=int,
                        help='number of runs to be done (parallel)')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    args = parser.parse_args()

    main(args)
