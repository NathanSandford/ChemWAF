import numpy as np
from scipy.stats import uniform, truncnorm

class UniformLogPrior:
    def __init__(self, label, lower_bound, upper_bound, out_of_bounds_val=-1e10):
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.out_of_bounds_val = out_of_bounds_val

    def __call__(self, x):
        return uniform.logpdf(x, loc=self.lower_bound, scale=self.upper_bound-self.lower_bound)
        #if (x < self.lower_bound) | (x > self.upper_bound):
        #    return self.out_of_bounds_val
        #else:
        #    return 0


class GaussianLogPrior:
    def __init__(self, label, mu, sigma, lower_bound=-np.inf, upper_bound=np.inf):
        self.label = label
        self.mu = mu
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = (lower_bound - mu) / sigma
        self.b = (upper_bound - mu) / sigma

    def __call__(self, x):
        return truncnorm.logpdf(x, a=self.a, b=self.b, loc=self.mu, scale=self.sigma)


class FlatLogPrior:
    def __init__(self, label):
        self.label = label

    def __call__(self, x):
        return np.zeros_like(x)
