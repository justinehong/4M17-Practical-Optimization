import numpy as np

class TabuSearch:
    """
    A class to represent the Tabu Search algorithm.

    Attributes
    ----------
    f : func
        objective function to be minimized
    x0 : array
        initial x
    DIM : int
        dimension of x
    seed : int
        random seed
    INTENSIFY : int
        counter threshold for search intensification
    DIVERSIFY : int
        counter threshold for search diversification
    REDUCE : int
        counter threshold for step size reduction
    counter : int
        counter controlling search stages
    LTMREGIONS : int
        number of regions of search space per variable for LTM
    XLIM : tuple(float, float)
        inequality constraints on each variable
    STMSIZE : int
        size of STM
    MTMSIZE : int
        size of MTM
    STM : np.array
        Short Term Memory
    MTM : np.array
        Medium Term Memory
    LTM : np.array
        Long Term Memory
    STEPRED : float
        factor for step size reduction
    TOL : float
        threshold for convergence
    INITIALSTEP : float
        initial step size
    step : float
        current step size
    ATOL : float
        tolerance for elementwise comparison between vectors (np.isclose)
    MAXFEVALS : int
        maximum number of objective function evaluations
    iters : int
        number of iterations performed
    fevals : int
        number of objective function evaluations performed
    base_history : list
        list of visited locations at each iteration
    f_history : list
        list of objective function values at each iteration
    best_fs : list
        list of best objective function values at each iteration
    """

    def __init__(self, f, x0, seed=1, intensify=10, diversify=15, reduce=25,
                 stmsize=7, mtmsize=4, xlim=(-512,512), maxfevals=15000,
                 initial_step=20, step_red=0.5, tol=0.001, atol=1e-05):
        """
        Class initialisation for Tabu Search algorithm.

        Parameters
        ----------
        f : func
            objective function to be minimized
        x0 : array
            initial x
        seed : int
            random seed
        intensify : int
            counter threshold for search intensification
        diversify : int
            counter threshold for search diversification
        reduce : int
            counter threshold for step size reduction
        stmsize : int
            size of STM
        mtmsize : int
            size of MTM
        xlim : tuple(float, float)
            inequality constraints on each variable
        maxfevals : int
            maximum number of objective function evaluations
        initial_step : float
            initial step size
        step_red : float
            factor for step size reduction
        tol : float
            threshold for convergence
        atol : float
            tolerance for elementwise comparison between vectors (np.isclose)
        """
        self.f = f
        self.x0 = np.array(x0)
        self.DIM = self.x0.shape[0]
        self.seed = seed
        self.RANDUNI = np.random.RandomState(seed=self.seed).uniform
        self.INTENSIFY = intensify
        self.DIVERSIFY = diversify
        self.REDUCE = reduce
        self.counter = 0
        self.LTMREGIONS = 2
        self.XLIM = xlim
        self.STMSIZE = stmsize
        self.MTMSIZE = mtmsize
        self.STM = None
        self.MTM = None
        self.LTM = np.zeros(self.LTMREGIONS**self.DIM)
        self.STEPRED = step_red
        self.TOL = tol
        self.INITIALSTEP = initial_step
        self.step = self.INITIALSTEP
        self.ATOL = atol
        self.MAXFEVALS = maxfevals
        self.iters = 0
        self.fevals = 0
        self.base_history = []
        self.f_history = []
        self.best_fs = []


    def set_x0(self, x0):
        """
        Set initial solution x0.

        Parameters
        ----------
        x0 -- initial solution
        """
        self.x0 = np.array(x0)


    def set_seed(self, seed):
        """
        Set seed for random number generator.
        
        Parameters
        ----------
        seed -- random number seed
        """
        self.seed = seed
        self.RANDUNI = np.random.RandomState(seed=self.seed).uniform


    def reset(self, x0):
        """
        Set all variables to initial values given initial x0.
        
        Parameters
        ----------
        x0 -- initial solution
        """
        self.counter = 0
        self.step = self.INITIALSTEP
        self.base_history = [x0]
        self.f_history = [self.f(x0)]
        self.best_fs = [self.f(x0)]
        self.RANDUNI = np.random.RandomState(seed=self.seed).uniform
        self.iters = 1
        self.fevals = 1
        self.STM = np.block([[x0.T]])
        self.MTM = np.block([[self.x0.T, self.f(self.x0)]])
        self.LTM = np.zeros(self.LTMREGIONS**self.DIM)
        self.update_LTM(x0)
        

    def check_convergence(self):
        """
        Check if step size is below threshold for convergence.
        
        Returns
        -------
        True if converged, False otherwise
        """
        if self.step < self.TOL:
            return True


    def local_search(self, base):
        """
        Perform a local search iteration:
        Starting from base, perform modified Hooke & Jeeves search.
        Perform pattern move if it improves the objective.

        Parameters
        ----------
        base -- current base location

        Returns
        -------
        Best move, objective value
        If there is no allowable move, return (None, None).
        """
        f_base = self.f(base)
        best_move = None
        f_best = np.inf
        x = base.copy()

        # For each variable in turn, increase/decrease by step
        for i in range(self.DIM):
            for inc in [-self.step, self.step]:
                x[i] += inc
                # If move is within bounds and not in STM
                if (x[i] >= self.XLIM[0] and x[i] <= self.XLIM[1] and
                    not np.isclose(x.T, self.STM, atol=self.ATOL).all(axis=1).any()):
                    f_new = self.f(x)
                    self.fevals += 1
                    # Accept if move decreases objective
                    if f_new < f_best:
                        best_move = x - base
                        f_best = f_new
                x[i] -= inc

        # If there is no allowable move, return None
        if best_move is None:
            return None, None
            
        # Perform pattern move and check if objective is lower
        if f_best < f_base:
            # Try performing two best moves
            x = np.add(x, best_move*2)
            # Check if move is valid
            if (x[i] >= self.XLIM[0] and x[i] <= self.XLIM[1] and
                    not np.isclose(x.T, self.STM, atol=self.ATOL).all(axis=1).any()):
                    self.fevals += 1
                    # Accept if pattern move decreases objective
                    if self.f(x) < f_best:
                        f_best = self.f(x)
            else:
                # Only perform one best move
                x = x - best_move
        else:
            x = x + best_move

        return x, f_best


    def update_STM(self, base):
        """
        Update the STM.
        
        Parameters
        ----------
        base -- current base location
        """
        # STM is not full
        if len(self.STM) < self.STMSIZE:
            self.STM = np.block([[self.STM],[base.T]])

        # STM is full
        else:
            self.STM = np.block([[self.STM[1:,:]],[base.T]])


    def update_MTM(self, base, f_base):
        """
        Update the MTM.
        MTM is sorted from lowest to highest objective value.

        Parameters
        ----------
        base -- current base location
        f_base -- objective function value at current location
        """
        # Check that base is not already in MTM
        row = np.block([[base.T, f_base]])
        if not np.isclose(self.MTM, row, atol=self.ATOL).all(axis=1).any():
            # Find first index of row that new solution has a higher objective than
            i = 0
            while i < len(self.MTM) and f_base < self.MTM[i][-1]:
                i += 1

            # MTM is not full
            if len(self.MTM) < self.MTMSIZE:
                self.MTM = np.block([[self.MTM[:i]],
                                    [base.T, f_base],
                                    [self.MTM[i:]]])
            
            # MTM is full and new solution is better than worst in MTM
            elif i > 0:
                self.MTM = np.block([[self.MTM[1:i]],
                                    [base.T, f_base],
                                    [self.MTM[i:]]])

    
    def update_LTM(self, base):
        """
        Update the LTM.
        Regions in the LTM are represented by binary numbers, with
        one bit corresponding to each variable.

        Parameters
        ----------
        base -- current base location
        """
        mid = (self.XLIM[0]+self.XLIM[1])/2
        # Binary list representing region for each dimension
        regions = np.array([0 if xi <= mid else 1 for xi in base])
        # Overall sector index converted from binary
        sector = 0
        for i, x in enumerate(reversed(regions)):
            sector += x * 2**i
        # Add one to count for that sector
        self.LTM[sector] += 1


    def update_counter(self, fval, verbose=False):
        """
        Update the counter controlling search intensification,
        serach diversification and step size reduction.
        Check if a new best solution has been found.

        Parameters
        ----------
        fval -- objective function value at current locaiton
        verbose -- True to print counter
        """
        # If new best solution is found, reset counter
        if fval < self.MTM[-1][-1]:
            self.counter = 0
        else:
            self.counter += 1
            if verbose:
                print("Counter: ", self.counter)


    def search_intensify(self):
        """
        Perform search intensification.
        Move search location to "average best", ie average of
        solutions in MTM.

        Returns
        -------
        New base, new objective value.
        """
        best_mean = np.mean(self.MTM[:,:-1], axis=0, keepdims=True).T
        return best_mean, self.f(best_mean)


    def search_diversify(self):
        """
        Perform search diversification.
        Move search location to a randomly selected location
        in the least explored region, as given by the counts
        in the LTM.

        Returns
        -------
        New base, new objective value.
        """
        # Find sector index with lowest counts (least explored)
        sector = np.argmin(self.LTM)
        # Get binary representation
        regions = bin(sector)[2:]
        # Pad with zeros
        while len(regions) < self.DIM:
            regions = '0' + regions
        regions = [int(x) for x in list(regions)]
        # Generate random vector within that region
        base = []
        mid = (self.XLIM[0]+self.XLIM[1])/2
        # Generate random value for each variable according to region
        for i in regions:
            if i == 0:
               base.append(self.RANDUNI(self.XLIM[0], mid)) 
            else:
                base.append(self.RANDUNI(mid, self.XLIM[1]))
        base = np.array(base).reshape((self.DIM,1))
        return base, self.f(base)


    def reduce_step(self):
        """
        Perform step size reduction.
        Move search location to the best solution found so far.

        Returns
        -------
        New base (best solution in MTM)
        """
        self.counter = 0
        self.step *= self.STEPRED
        return self.MTM[-1][:-1]


    def run(self, n_iters, verbose=False):
        """
        Run the Tabusearch algorithm.

        Parameters
        ----------
        n_iters -- max number of search iterations
        verbose -- True to print search progress

        Returns
        -------
        Best solution location, corresponding objective, list
        of all locations visited and corresponding objectives,
        list of best objective values at each iteration
        """
        self.reset(self.x0)
        base = self.x0.copy()

        while self.iters < n_iters and self.fevals < self.MAXFEVALS:
            if verbose:
                print("i", self.iters, base, self.f_history[-1])
            if self.counter == self.REDUCE:
                if verbose:
                    print("Reduce")
                base = self.reduce_step()
                if self.check_convergence():
                    break
            else:
                if self.counter == self.INTENSIFY:
                    if verbose:
                        print("Intensify")
                    base, fx = self.search_intensify()
                elif self.counter == self.DIVERSIFY:
                    if verbose:
                        print("Diversify")
                    base, fx = self.search_diversify()
                else:
                    x, fx = self.local_search(base)
                    # If there is no allowable move
                    if x is None:
                        if verbose:
                            print("No allowable move", base)
                        # Diversify search
                        x, fx = self.search_diversify()
                    base = x.copy()

                self.update_counter(fx, verbose)
                self.update_STM(base)
                self.update_MTM(base, fx)
                self.update_LTM(base)
                self.base_history.append(base)
                self.f_history.append(fx)
                self.best_fs.append(self.MTM[-1][-1])
                self.iters += 1

        return (self.MTM[-1][:-1], self.MTM[-1][-1],
                self.base_history, self.f_history, self.best_fs)
                
        

def evaluate_performance(tabu, dim, nruns=25, numiters=1000):
    """
    Get performance metrics for Tabu Search with given model parameters.

    Parameters
    ----------
    tabu -- TabuSearch object
    dim -- number of dimensions of space
    nruns -- number of runs to perform with different random number sequences
    numiters -- max number of iterations for each run

    Returns
    -------
    Mean best objective and standard deviation of best objectives
    found over all runs, mean number of iterations to convergence,
    mean number of objective function evaluations.
    """
    objs = []
    its = []
    fevals = []
    for seed in range(nruns):
        np.random.seed(seed)
        x0 = np.random.uniform(-512,512,(dim,1))
        tabu.set_x0(x0)
        tabu.set_seed(seed)
        sol, fsol, xs, fs, bestfs = tabu.run(numiters)
        objs.append(bestfs[-1])
        its.append(tabu.iters)
        fevals.append(tabu.fevals)
    objs = np.array(objs)
    its = np.array(its)
    fevals = np.array(fevals)
    return (objs.mean(), objs.std(), its.mean(), fevals.mean())