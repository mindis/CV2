import random

def ransac(sample_pop, algorithm, error_function, n_samples=8, n_iter=100,
        additional_args=None):
    
    smallest_err = float('inf')
    best = None
    if n_samples > len(sample_pop):
        raise Exception('''It is impossible to sample %d samples from 
                population with length %d''' % (n_samples, len(sample_pop)))

    for _ in xrange(n_iter):
        sample = random.sample(sample_pop, n_samples)
        result = algorithm(sample, **additional_args)
        err = 0
        for points in sample_pop:
            err += error_function(points, result)

        if err < smallest_err:
            smallest_err = err
            best = result
    
    return best

