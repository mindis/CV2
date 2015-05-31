import random
import numpy as np


def get_error(error_function, model, test_points):
    test_err = np.zeros(test_points.shape[0])
    for i, points in enumerate(test_points):
        test_err[i] = error_function(points, model)
    return test_err


# lo-ransac
# todo: add converge detection
def ransac(sample_pop, algorithm, error_function, n_samples=8, n_iter=5000,
           t=1e-4, acceptance=0.25, verbose=True,
           additional_args=None):
    # modified to reflect http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    sample_pop = np.array(sample_pop)
    bestfit = None
    best_sample = None
    besterr = -np.inf
    converge_counter = 0
    
    if n_samples > len(sample_pop):
        raise Exception('''It is impossible to sample %d samples from 
                population with length %d''' % (n_samples, len(sample_pop)))

    for i in range(1,n_iter+1):
        maybe_idx = np.random.choice(len(sample_pop), size=n_samples)
        maybeinliers = sample_pop[maybe_idx] 

        # Generate a model and calculate the error
        maybemodel = algorithm(maybeinliers, **additional_args)
        test_err = get_error(error_function, maybemodel, sample_pop)
        inlier_idx = np.where(test_err < t)[0]   # select indices of rows with accepted points

        # Check model acceptance
        if len(inlier_idx) > acceptance * len(sample_pop):
            better_data = sample_pop[inlier_idx]
            bettermodel = algorithm(better_data, **additional_args)
            better_errs = get_error(error_function, bettermodel, better_data)
            better_inliers_idx = np.where(better_errs < t)[0]

            if len(better_inliers_idx) > besterr:
                bestfit = bettermodel
                besterr = len(better_inliers_idx)
                best_sample = better_data[better_inliers_idx]

                if verbose:
                    print 'iteration %d - best inlier %d' % (i, besterr)

                # reset counter
                converge_counter = 0

        converge_counter += 1
        if converge_counter > 1000:
            print "converged.."
            break

    return bestfit, best_sample
