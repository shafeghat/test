"""
Functions for computing coordinate transformations, derivatives and
sample means of parameters and data.

---
hi, this is Nasrin's editing
State-Space Analysis of Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012)
Copyright (C) 2014  Thomas Sharp (thomas.sharp@riken.jp)
"""
import itertools
import numpy
import pdb
import scipy.misc



# Matrix to map from theta to probability
p_map = None
# Matrix to map from probability to eta
eta_map = None



def compute_D(N, O):
    """
    Computes the number of natural parameters for the given number of cells and
    order of interactions.

    :param int N:
        Total cells in the spike data.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...

    :returns:
        Number of natural parameters for the spike-pattern interactions.
    """
    D = numpy.sum([scipy.misc.comb(N, k, exact=1) for k in xrange(1, O + 1)])

    return D


def compute_eta(p):
    """
    Computes the expected values, eta, of spike patterns

        n_0,1 = p(0,1) + p(1,1) # for example

    from the supplied probability mass.

    :param numpy.ndarray p:
        Probability mass of spike patterns.

    :returns:
        Expected values of spike patterns (eta) as a numpy.ndarray.
    """
    global eta_map

    eta = numpy.dot(eta_map, p)

    return eta


def compute_fisher_info(p, eta):
    """
    Computes the Fisher-information matrix of the expected values, eta, of spike
    patterns for the purposes of Newton-Raphson gradient-ascent and
    computation of the marginal probability distribution. For example, for two
    neurons:

        H = [n1 - n1^2,      n12 - n1 * n2,  n12 - n1 * n12,
             n12 - n1 * n2,  n2 - n2^2,      n12 - n2 * n12,
             n12 - n1 * n12, n12 - n2 * n12, n12 - n12^2]

    :param numpy.ndarray p:
        Probability mass of spike patterns.
    :param numpy.ndarray eta:
        Expected values of spike patterns.

    :returns:
        Fisher-information matrix as a numpy.ndarray.
    """
    global p_map, eta_map

    # Stack columns of p for next step
    p_stack = numpy.repeat(p, eta.size).reshape(p.size, eta.size)
    # Compute Fisher matrix
    fisher = numpy.dot(eta_map, p_stack * p_map) - numpy.outer(eta, eta)

    return fisher


def compute_p(theta):
    """
    Computes the probability distribution of spike patterns, for example

        p(x1,x2) =     e^(t1x1)e^(t2x2)e^(t12x1x2)
                   -----------------------------------
                   1 + e^(t1) + e^(t2) + e^(t1+t2+t12)

    from the supplied natural parameters.

    :param numpy.ndarray theta:
        Natural `theta' parameters: t1, t2, ..., t12, ...

    :returns:
        Probability mass as a numpy.ndarray.
    """
    global p_map

    # Compute log probabilities
    log_p = numpy.dot(p_map, theta)
    # Take exponential and normalise
    p = numpy.exp(log_p)
    p_tmp = p / numpy.sum(p)

    return p_tmp


def compute_psi(theta):
    """
    Computes the normalisation parameter, psi, for the log-linear probability
    mass function of spike patterns. For example, for two neurons

        psi(theta) = 1 + e^(t1) + e^(t2) + e^(t1+t2+t12)

    :param numpy.ndarray theta:
        Natural `theta' parameters: t1, t2, ..., t12, ...

    :returns:
        Normalisation parameter, psi, of the log linear model as a float.
    """
    global p_map

    # Take coincident-pattern subsets of theta
    tmp = numpy.dot(p_map, theta)
    # Take the sum of the exponentials and take the log
    tmp = numpy.sum(numpy.exp(tmp))
    psi = numpy.log(tmp)

    return psi


def compute_y(spikes, order, window):
    """
    Computes the empirical mean rate of each spike pattern across trials for
    each timestep up to `order', for example

        y_12,t = 1 / N * \sigma^{L} X1_l,t * X2_l,t

    is a second-order pattern where t is the timestep and l is the trial.

    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    :param int window:
        Bin-width for counting spikes, in milliseconds.

    :returns:
        Trial-mean rates of each pattern in each timestep, as a numpy.ndarray
        with `time' rows and sum_{k=1}^{order} {n \choose k} columns, given
        n cells.
    """
    # Get spike-matrix metadata
    T, R, N = spikes.shape
    # Bin spikes
    spikes = spikes.reshape((T / window, window, R, N))
    spikes = spikes.any(axis=1)
    # Compute each n-choose-k subset of cell IDs up to `order'
    subsets = enumerate_subsets(N, order)
    # Set up the output array
    y = numpy.zeros((T / window, len(subsets)))
    # Iterate over each subset
    for i in xrange(len(subsets)):
        # Select the cells that are in the subset
        sp = spikes[:,:,subsets[i]]
        # Find the timesteps in which all subset-cells spike coincidentally
        spc = sp.sum(axis=2) == len(subsets[i])
        # Average the occurences of coincident patterns to get the mean rate
        y[:,i] = spc.mean(axis=1)

    return y


def enumerate_subsets(N, O):
    """
    Enumerates all N-choose-k subsets of cell IDs for k = 1, 2, ..., O. For
    example,

        >>> compute_subsets(4, 2)
        >>> [(0,), (1,), (2,), (3,), (0, 1),
                (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    :param int N:
        Total cells from which to choose subsets.
    :param int O:
        Maximum size of subsets to enumerate.

    :returns:
        List of tuples, each tuple containing a subset of cell IDs.
    """
    # Compute each C-choose-k subset of cell IDs up to `O'
    subsets = list()
    ids = numpy.arange(N)
    for k in xrange(1, O + 1):
        subsets.extend(list(itertools.combinations(ids, k)))
    # Assert that we've got the correct number of subsets
    assert len(subsets) == compute_D(N, O)

    return subsets


def enumerate_patterns(N):
    """
    Enumerates all spike patterns in order, for example:

        >>> enumerate_patterns(3)
        array([[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [1, 1, 1]], dtype=uint8)

    :param int N:
        Number of cells for which to enumerate patterns.

    :returns:
        Binary matrix of spike patterns as a numpy.ndarray of dimensions
        (2**N, N).
    """
    # Get the spike patterns as ordered subsets
    subsets = enumerate_subsets(N, N)
    assert len(subsets) == 2**N - 1
    # Generate output array and fill according to subsets
    fx = numpy.zeros((2**N, N), dtype=numpy.uint8)
    for i in xrange(len(subsets)):
        fx[i+1,subsets[i]] = 1

    return fx


def initialise(N, O):
    """
    Sets up matrices to transform between theta, probability and eta
    coordinates. Computing probability requires finding subsets of theta for the
    numerator; for example, with two cells, finding the numerator:

        p(x1,x2) =     e^(t1x1)e^(t2x2)e^(t12x1x2)
                   -----------------------------------
                   1 + e^(t1) + e^(t2) + e^(t1+t2+t12)

    We calculate a `p_map' to do this for arbitrary numbers of neurons and
    orders of interactions. To compute from probabilities to eta coordinates,
    we use an `eta_map' that is just the transpose of the `p_map'. The method
    for doing this is a bit tricky, and not easy to explain. Suffice to say,
    it produces to the correct maps.

    This function has the side effect of setting global variables `p_map' and
    `eta_map', which are used later by other functions in this module.

    :param int N:
        Total cells in the spike data.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    """
    global p_map, eta_map

    # Create a matrix of binary spike patterns
    fx = enumerate_patterns(N)
    # Compute the number of natural parameters, given the order parameter
    D = compute_D(N, O)
    # Set up the output matrix
    p_map = numpy.ones((2**N, D), dtype=numpy.uint8)
    # Compute the map!
    for i in xrange(1, D+1):
        idx = numpy.nonzero(fx[i,:])[0]
        for j in xrange(idx.size):
            p_map[:,i-1] = p_map[:,i-1] & fx[:,idx[j]]
    # Set up the eta map
    eta_map = p_map.transpose()
