import random
import numpy as np


def get_error(error_function, model, test_points):
    test_err = np.array([])
    for points in test_points:
        test_err = np.append(test_err, error_function(points, model))

    return test_err


# lo-ransac
def ransac(sample_pop, algorithm, error_function, n_samples=8, n_iter=5000,
           t=1E-6, acceptance=0.25, verbose=False,
           additional_args=None):
    # modified to reflect http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    i = 0
    d = acceptance * len(sample_pop)
    sample_pop = np.array(sample_pop)
    bestfit = None
    best_inlier_idxs = []
    besterr = np.inf
    if n_samples > len(sample_pop):
        raise Exception('''It is impossible to sample %d samples from 
                population with length %d''' % (n_samples, len(sample_pop)))

    while i < n_iter:
        maybe_idxs, test_idxs = random_partition(n_samples, len(sample_pop))
        maybeinliers = sample_pop[maybe_idxs]
        test_points = sample_pop[test_idxs]

        maybemodel = algorithm(maybeinliers, **additional_args)
        test_err = get_error(error_function, maybemodel, test_points)
        also_idxs = test_idxs[test_err < t]  # select indices of rows with accepted points
        alsoinliers = sample_pop[also_idxs]

        if len(alsoinliers) > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = algorithm(betterdata, **additional_args)
            better_errs = get_error(error_function, bettermodel, betterdata)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

            if verbose:
                print 'iteration %d - best inlier %d' % (i, len(best_inlier_idxs))
        i += 1

    return bestfit, sample_pop[best_inlier_idxs]


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2