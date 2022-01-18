import numpy as np

class SimulatedAnnealing:
    """
    A class to represent the Simulated Annealing algorithm.

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
    XLIM : tuple(float, float)
        inequality constraints on each variable
    PROPOSAL : str
        proposal method for generating trial solutions ('simple', 'parks')
    ALPHA : float
        damping constant for 'parks' method
    OMEGA : float
       update weighting for 'parks' method
    DLIM : tuple(float, float)
        max step size in update matrix D
    INITIAL_STEP : float
        initial step size
    D : np.array
        step size matrix
    INITIAL_T_METHOD : str
        method for determining initial temperature ('kirkpatrick', 'white')
    INITIAL_T_PROB: float
        average probability of acceptance at initial temperature
    INITIAL_T : float
        initial tempertature
    T : float
        current temperature
    L_K : int
        maximum Markov chain length
    ETA_MIN : int
        minimum acceptances per chain
    ANNEALING : str
        annealing method ('kirkpatrick', 'huang')
    ANNEAL_FACTOR : float
        exponential cooling factor for 'kirkpatrick' method
    RESTART : int
        number of iterations with no new best solution to trigger restart
    MAXFEVALS : int
        maximum number of objective function evaluations
    MIN_ACC : float
        minimum solution acceptance ratio for termination
    iters : int
        number of iterations performed
    fevals : int
        number of objective function evaluations performed
    x_history : list
        list of visited locations at each iteration
    f_history : list
        list of objective function values at each iteration
    best_fs : list
        list of best objective function values at each iteration
    best_solution : array
        best solution
    best_counter : int
        counter for iterations since new best found
    """

    def __init__(self, f, x0, seed=1, xlim=(-512,512), initial_step=20,
                 maxfevals=15000, proposal='simple', alpha=0.1, omega=2.1,
                 initial_t_method='kirkpatrick', initial_t_prob=0.8, l_k=100,
                 eta_min=0.6, Dlim = (-300,300), annealing='kirkpatrick',
                 anneal_factor=0.9, restart=500, min_acc=0.1):
        """
        Class initialisation for Simulated Annealing algorithm.

        Parameters
        ----------
        f : func
            objective function to be minimized
        x0 : array
            initial x
        seed : int
            random seed
        xlim : tuple(float, float)
            inequality constraints on each variable
        initial_step : float
            initial step size
        maxfevals : int
            maximum number of objective function evaluations
        proposal : str
            proposal method for generating trial solutions ('simple', 'parks')
        alpha : float
            damping constant for 'parks' method
        omega : float
            update weighting for 'parks' method
        initial_t_method : str
            method for determining initial temperature ('kirkpatrick', 'white')
        initial_t_prob: float
            average probability of acceptance at initial temperature
        l_k : int
            maximum Markov chain length
        eta_min : int
            minimum acceptances per chain
        Dlim : tuple(float, float)
            max step size in update matrix D
        annealing : str
            annealing method ('kirkpatrick', 'huang')
        anneal_factor : float
            exponential cooling factor for 'kirkpatrick' method
        restart : int
            number of iterations with no new best solution to trigger restart
        min_acc : float
            minimum solution acceptance ratio for termination
        """
        self.f = f
        self.x0 = np.array(x0)
        self.DIM = self.x0.shape[0]
        self.seed = seed
        self.RANDUNI = np.random.RandomState(seed=self.seed).uniform
        self.XLIM = xlim
        self.PROPOSAL = proposal                       # 'simple', 'parks'
        self.ALPHA = alpha
        self.OMEGA = omega
        self.DLIM = Dlim
        self.INITIAL_STEP = initial_step
        self.D = np.eye(self.DIM) * self.INITIAL_STEP
        self.INITIAL_T_METHOD = initial_t_method        # 'kirkpatrick', 'white'
        self.INITIAL_T_PROB = initial_t_prob
        self.INITIAL_T = None
        self.T = None
        self.L_K = l_k
        self.ETA_MIN = eta_min
        self.ANNEALING = annealing                      # 'kirkpatrick', 'huang'
        self.ANNEAL_FACTOR = anneal_factor
        self.RESTART = restart
        self.MAXFEVALS = maxfevals
        self.MIN_ACC = min_acc
        self.iters = 0
        self.fevals = 0
        self.x_history = []
        self.f_history = []
        self.best_fs = []
        self.best_x = self.x0
        self.best_counter = 0

        # Initialise temperature
        self.initialise_temp()


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
        self.x_history = [x0]
        self.f_history = [self.f(x0)]
        self.best_fs = [self.f(x0)]
        self.best_x = x0
        self.RANDUNI = np.random.RandomState(seed=self.seed).uniform
        self.iters = 1
        self.fevals = 1
        self.best_counter = 0
        self.D = np.eye(self.DIM) * self.INITIAL_STEP
        self.T = self.INITIAL_T


    def initialise_temp(self):
        """
        Initialise temperature using method defined by self.PROPOSAL.
        Perform initial search and accept all moves.

        Kirkpatrick: average probability of solution that
        increases objective being accepted of self.INITIAL_T_PROB
        White: std deviation of variation in objective function.
        """
        n_trials = 300
        dfs = []
        obj_vars = []
        x = self.x0.copy()
        f_prev = self.f(x)
        for i in range(n_trials):
            u = self.RANDUNI(-1, 1, (self.DIM, 1))
            r = self.D @ u
            x_new = x + r
            # Limit solution within inequality constraints
            x_new = np.maximum(x_new, [self.XLIM[0]])
            x_new = np.minimum(x_new, [self.XLIM[1]])
            f_new = self.f(x_new)
            if f_new > f_prev:
                dfs.append(f_new - f_prev)
            elif self.PROPOSAL == 'parks':
                # Update proposal matrix
                self.D = (1 - self.ALPHA) * self.D + self.ALPHA * self.OMEGA * np.diag(np.abs(r.flatten()))
                self.D = np.maximum(self.D, self.DLIM[0])
                self.D = np.minimum(self.D, self.DLIM[1])

            obj_vars.append(f_new - f_prev)
            x = x_new
            f_prev = f_new
        
        if self.INITIAL_T_METHOD == 'kirkpatrick':
            av_inc = np.array(dfs).mean()
            self.INITIAL_T = -av_inc/np.log(self.INITIAL_T_PROB)
        elif self.INITIAL_T_METHOD == 'white':
            self.INITIAL_T = np.array(obj_vars).std()
        print("Initial T: ", self.INITIAL_T)


    def cool(self, sigma_k):
        """
        Reduce temperature using annealing method.

        Kirkpatrick: exponential cooling.
        Huang: adaptive cooling.

        Parameters
        ----------
        sigma_k -- standard deviation of objective function values
        accepted at temperature T_k
        """
        if self.ANNEALING == "kirkpatrick":
            self.T *= self.ANNEAL_FACTOR
        elif self.ANNEALING == "huang":
            self.T *= max(0.5, np.exp(-0.7*self.T/sigma_k))


    def check_restart(self, f):
        """
        Check if condition for restart has been met. 
        
        Parameters
        ----------
        f -- objective function value at current location

        Returns
        -------
        True if condition for restart has been met, False otherwise
        """
        if f > self.best_fs[-1]:
            self.best_counter += 1
        else:
            self.best_counter = 0
        if self.best_counter == self.RESTART:
            return True
        else:
            return False
        

    def run(self, n_iters, verbose=False):
        """
        Run the Simulated Annealing algorithm.

        Parameters
        ----------
        n_iters -- max number of search iterations
        verbose -- True to print progress

        Returns
        -------
        Best solution and objective, list of all locations visited,
        corresponding objectives, list of best objective values at
        each iteration.
        """
        self.reset(self.x0)
        x = self.x0.copy()
        f_prev = self.f(x)
        L = 0               # current chain length
        acc_counter = 0     # number of acceptances in current chain
        f_accs = []         # objective function values for each acceptance
        acc_ratio = 1       # acceptance ratio
            
        while self.iters < n_iters and self.fevals < self.MAXFEVALS:
            if verbose:
                print("i", self.iters, x.flatten(), self.f_history[-1])

            # Reset chain and decrement temperature if threshold reached
            if L == self.L_K or acc_counter == self.ETA_MIN:
                sigma_k = np.array(f_accs).std()
                self.cool(sigma_k)
                L = 0
                acc_counter = 0
                f_accs = []
                if verbose:
                    print("Cooling to ", self.T)
                if acc_ratio < self.MIN_ACC:
                    if verbose:
                        print("Acceptance ratio too low")
                    break

            # Generate random update vector and new solution
            u = self.RANDUNI(-1, 1, (self.DIM, 1))
            r = self.D @ u
            x_new = x + r
            # Limit solution within inequality constraints
            x_new = np.maximum(x_new, [self.XLIM[0]])
            x_new = np.minimum(x_new, [self.XLIM[1]])
            f_new = self.f(x_new)
            self.fevals += 1
            if verbose:
                print("Propose: ", x_new.flatten(), f_new)
            df = f_new - f_prev
            # Calculate step size
            d_bar = 1
            if self.PROPOSAL == 'parks':
                R = np.diag(np.abs(r.flatten()))
                d_bar = np.sqrt(np.sum(np.square(r)))
            # Calculate probability of acceptance
            p = np.exp(-df/(self.T * d_bar))
            sample = self.RANDUNI(0,1)
            if sample <= p:
                # Update x if accepted
                acc_counter += 1
                f_accs.append(f_new)
                x = x_new.copy()
                f_prev = f_new
                if self.PROPOSAL == 'parks':
                    # Update proposal matrix
                    self.D = (1 - self.ALPHA) * self.D + self.ALPHA * self.OMEGA * R
                    self.D = np.maximum(self.D, self.DLIM[0])
                    self.D = np.minimum(self.D, self.DLIM[1])
                if verbose:
                    print("Accept", sample, p)
            elif verbose:
                print("Reject", sample, p)
           
            # Check if restart condition is met
            if self.check_restart(f_prev):
                if verbose:
                    print("Restart", self.best_x)
                x = self.best_x

            # Update best location found
            if f_prev < self.best_fs[-1]:
                self.best_x = x

            L += 1
            self.iters += 1
            self.x_history.append(x)
            self.f_history.append(f_prev)
            self.best_fs.append(min(self.best_fs[-1], f_prev))
            acc_ratio = acc_counter/L

        return (self.best_x, self.best_fs[-1], self.x_history, self.f_history, self.best_fs)


def evaluate_performance(sa, dim, nruns=25, numiters=1000):
    """
    Get performance metrics for Simulated Annealing with given model parameters.

    Parameters
    ----------
    sa -- SimulatedAnnealing object
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
        sa.set_x0(x0)
        sa.set_seed(seed)
        sol, fsol, xs, bestfs = sa.run(numiters)
        objs.append(bestfs[-1])
        its.append(sa.iters)
        fevals.append(sa.fevals)
    objs = np.array(objs)
    its = np.array(its)
    fevals = np.array(fevals)
    return objs.mean(), objs.std(), its.mean(), fevals.mean()