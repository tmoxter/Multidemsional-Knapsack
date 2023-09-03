import numpy as np
import time
from copy import deepcopy
from utils import RepairOperator

class SimpleGA:
    """Simple genetic algorithm for the mkp using uniform crossover and mutation.

    Parameters
    ----
    shape : tuple, dimesionality of the problem and the population
    problem : object, an MKProblem instance
    use_repair : bool, whether to use the repair operator (always used for initilization)
    check_diversity : bool, whether to terminate once diversity is lost,
    max_evals : int, maximum number of evaluatios,
    max_gens : int, maximum number of generations,
    max_time : float, maximum runtime,
    verbose : bool, print run statistics

    Attributes
    ----
    num_gens : int: number of performed generations,
    num_evals : int, number of performed evaluatins,
    best_of_gens : list, elitist solutions per generation,
    runData : list, dict per generation with performancen data
    """
    
    def __init__(self, shape : tuple, problem : object,
    use_repair : bool = True,
    check_diversity : bool = False, vtr : float = None,
    max_evals : int=None, max_gens : int=100, max_time : int=None,
    verbose : bool=False, log = None) -> None:

        self.shape = shape
        self.problem = problem
        self.use_repair = use_repair
        self.vtr = vtr
        self.max_evals = max_evals
        self.max_gens = max_gens
        self.max_time = max_time
        self.check_diversity = check_diversity
        self.verbose = verbose
        self.log = log
        
        self._repair = RepairOperator(problem)
        self.num_gens = 0
        self.num_evals = 0
        self.best_of_gens = list()
        self.runData = []
    
    def _initialize_population(self):
        """If no population is provided, randomly initialize
        and repair"""

        self.population = np.random.rand(self.shape).round()
        self._repair(self.population)
        self.gen_fitness = self.problem.fitness(self.population)
    
    def _must_terminate(self) -> bool:
        """Check if termination must occur."""

        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        elif all([(self.population[0] == o).all() for o in self.population])\
                    and self.pop_convergence:
            return True
        elif max(self.gen_fitness) == self.vtr:
            return True
        return False

    def _generational_step(self) -> None:
        """Evolve one generation."""

        # --- variation & selection ---
        offspring = self._uniform_crossover(self.population)
        self._mutation(offspring)
        if self.use_repair:
            self._repair(offspring)
        self._selection(offspring, include_parents = True)
        self.gen_fitness = self.problem.fitness(self.population)

        # --- update info ---
        self.num_gens += 1
        self.num_evals += self.shape[0]
        best = self.population[np.argmax(self.fitness(self.population))]
        self.best_of_gens.append(best)

    def _selection(self, offspring : np.ndarray, include_parents : bool,
                    tournament_size : int = 2) -> np.ndarray:
        """Tournament selection"""

        if include_parents:
            contestants = np.append(self.population, offspring)
            fitnesses = self.gen_fitness+self.fitness(contestants)
        else:
            contestants = offspring
            fitnesses = self.fitness(contestants)
        n_to_select = self.shape[0]
        n = contestants.shape[0]
        n_per_parse = n // tournament_size
        n_parses = n_to_select // n_per_parse

        # --- assert quantities are compatible ---
        assert n / tournament_size == n_per_parse,\
            "Number of contestants {} is not a multiple of tournament size {}"\
                                                    .format(n, tournament_size)
        assert n_to_select / n_per_parse == n_parses
        
        select = np.zeros(n_to_select)
        permute = np.arange(0, n)
        for i in range(n_parses):
            np.random.permutation(permute)
            contestants = contestants[permute,:]
            fitnesses = fitnesses[permute,:]
            select[i*n_per_parse:(i+1)*n_per_parse] = np.argmax(
                fitnesses.reshape((-1, tournament_size)), axis=1)\
                + np.arange(0, n, tournament_size)
        
        self.population[:,:] = contestants[select]
    
    def _uniform_crossover(self, population) -> np.ndarray:
        """Uniform crossover operator"""

        offspring = np.zeros_like(population)
        pairs = np.random.permutation(
            np.arange(population.shape[0])).reshape(2,-1)
        for i, pair in enumerate(pairs):
            mask = np.random.rand(population.shape[0]).round()
            offspring[i] = np.where(mask, population[pair[0]], population[pair[1]])
            offspring[i+1] = np.where(mask, population[pair[1]], population[pair[0]])
        
        return offspring
    
    def _mutation(self, population, chance = None):
        """
        Mutation operator, performs random bit flips.
        """

        idx = np.random.rand(self.shape)
        if not chance:
            chance = 2/self.shape[1]
        if self.use_repair:
            population[np.where(idx < chance)] *=-1
        else:
            recover = deepcopy(population)
            population[np.where(idx < chance)] *=-1
            valid = self.constraint(population)
            population = np.where(~valid, recover, population)         

        return population
    
    def evolve(self, population = None) -> None:
        """Start the evolution process. Provide a population, e.g. to
        continue search or let it be newly initialized"""

        if not population:
            self.population = self._initialize_population()
        else:
            self.populaiton = population

        self.start_time = time.time()
        self.runData.append({"n_eval":self.num_evals,
                             "max_f":max(self.fitness(self.population))})

        # --- generational loop ---
        while not self._must_terminate():
            # --- --- perform one generation --- ---
            self._generational_step()
            # --- --- logging & printing --- ---
            if self.log:
                self.log.add_scalar("Best / Gen", self.fitness(self.best_of_gens[-1]), self.num_gens)
                self.log.add_scalar("Pop Fitness / Gen", sum(self.gen_fitness), self.num_gens)
                      
            if self.verbose:
                print("Gen: {}, best of gen fitness: {:.3f}".format(
                    self.num_gens, float(self.fitness(self.best_of_gens[-1])))
                )

            self.runData.append(
                {"n_eval":self.num_evals, "max_f":self.fitness(self.best_of_gens[-1])}
            )
        
        print("Best found f: {}".format(self.fitness(self.best_of_gens[-1])))
    

class HillClimber:
    """Hill climbing algorithm for the mkp using local search with
    a population of solution candidates.

    Parameters
    ----
    shape : tuple, dimesionality of the problem and the population
    problem : object, an MKProblem instance
    use_repair : bool, whether to use the repair operator (always used for initilization)
    n_bits : int, number of bits to flip per local search step,
    max_evals : int, maximum number of evaluatios,
    max_gens : int, maximum number of generations,
    max_time : float, maximum runtime,
    verbose : bool, print run statistics

    Attributes
    ----
    num_gens : int: number of performed generations,
    num_evals : int, number of performed evaluatins,
    best_of_gens : list, elitist solutions per generation,
    runData : list, dict per generation with performancen data

    """
    def __init__(self, shape : tuple, problem : object,
    use_repair : bool = True, n_bits : int = 1,
    max_evals : int=None, max_gens : int=100, max_time : int=None,
    vtr : float = None,
    verbose : bool=False, log = None) -> None:

        self.shape = shape
        self.problem = problem
        self.use_repair = use_repair
        self.max_evals = max_evals
        self.max_gens = max_gens
        self.max_time = max_time
        self.vtr = vtr
        self.verbose = verbose
        self.n_bits = n_bits
        self.log = log

        self._repair = RepairOperator(problem)
        self.num_gens = 0
        self.num_evals = 0
        self.best_of_gens = list()
        self.runData = []
        self.gen_fitness = self.fitness(self.population)
    
    def _initialize_population(self):
        self.population = np.random.rand(self.shape).round()
        self._repair(self.population)
        self.gen_fitness = self.problem.fitness(self.population)
    
    def _must_terminate(self) -> bool:
        """Check if termination must occur."""

        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        elif self.vtr and max(self.gen_fitness) == self.vtr:
            return True
        return False

    def _step(self) -> None:
        """Perform local variation on population without any
        domain specific operations."""

        recover = deepcopy(self.population)
        for _ in range(self.n_bits):
            idx = np.random.randint(0, self.shape[1],(self.shape[0], ))
            self.population[np.arange(self.shape[0]),idx] *=-1
        if self.use_repair:
            self.population = self._repair(self.population)
        else:
            valid = self.constraint(self.population)
            self.population[np.where(~valid)[0]] = recover[np.where(~valid)[0]]
        idxs_to_rec = self.fitness(self.population)<self.gen_fitness
        self.population[idxs_to_rec] = recover[idxs_to_rec]

    def climb(self, population = None) -> None:

        if not population:
            self.population = self._initialize_population()
        else:
            self.populaiton = population
        self.start_time = time.time()
        self.runData.append({"n_eval":self.num_evals,
                             "max_f":max(self.fitness(self.population))})

        # --- generational loop ---
        while not self._must_terminate():
            # --- --- perform one generation --- ---
            self._step()
            self.gen_fitness = self.problem.fitness(self.population)

            # --- --- update info & logging --- ---
            self.num_gens += 1
            self.num_evals += self.shape[0]
            best = self.population[np.argmax(self.fitness(self.population))]
            self.best_of_gens.append(best)

            if self.log:
                self.log.add_scalar("Best / Gen", self.fitness(self.best_of_gens[-1]), self.num_gens)
                self.log.add_scalar("Pop Fitness / Gen", sum(self.gen_fitness), self.num_gens)
            
            if self.verbose:
                print("Gen: {}, best of gen fitness: {:.3f}".format(
                    self.num_gens, float(self.fitness(self.best_of_gens[-1])))
                )
            
            self.runData.append(
                {"n_eval":self.num_evals, "max_f":self.fitness(self.best_of_gens[-1])}
            )

    def check_convergence(self):
        """Check if the algorithm has converged to a single solution."""
        return len(np.unique(self.population, axis=0)) == 1