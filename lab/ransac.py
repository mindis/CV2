import random
import numpy as np


def get_error(error_function, model, test_points):
    test_err = np.zeros(test_points.shape[0])
    for i, points in enumerate(test_points):
        test_err[i] = error_function(points, model)
    return test_err


# lo-ransac
def ransac(sample_pop, algorithm, error_function, n_samples=8, n_iter=5000,
           t=1E-3, acceptance=0.25, verbose=True,
           additional_args=None):
    # modified to reflect http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    sample_pop = np.array(sample_pop)
    bestfit = None
    besterr = -np.inf
    
    if n_samples > len(sample_pop):
        raise Exception('''It is impossible to sample %d samples from 
                population with length %d''' % (n_samples, len(sample_pop)))

    for i in range(1,n_iter+1):
        maybeinliers, test_points = divide_sample(sample_pop, n_samples)
        
        #Generate a model and calculate the error
        maybemodel = algorithm(maybeinliers, **additional_args)
        test_err = get_error(error_function, maybemodel, test_points)
        also_idxs = np.where(test_err < t)[0]   # select indices of rows with accepted points
        alsoinliers = sample_pop[also_idxs]     # find the inliers of this model

        # Check model acceptance
        if len(alsoinliers) > acceptance * len(sample_pop):


            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = algorithm(betterdata, **additional_args)
            better_errs = get_error(error_function, bettermodel, betterdata)
            betterinliers = betterdata[better_errs < t]

            if betterinliers.shape[0] > besterr:
                bestfit = bettermodel
                besterr = betterinliers.shape[0]
                best_sample = np.concatenate((maybeinliers, alsoinliers))

                if verbose:
                    print 'iteration %d - best inlier %d' % (i, besterr)

    return bestfit, best_sample

#Split sample into points to use in algorithm and points to test error on.
def divide_sample(sample_pop, n_samples):
    maybe_idxs = np.random.choice(len(sample_pop), size=n_samples)
    bool_idx = np.zeros((sample_pop.shape[0]), dtype=bool)
    bool_idx[maybe_idxs] = True
    maybeinliers = sample_pop[bool_idx]
    test_points = sample_pop[~bool_idx]

    return maybeinliers, test_points


