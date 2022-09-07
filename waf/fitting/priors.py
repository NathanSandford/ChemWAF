import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import uniform, truncnorm, rv_histogram
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
from waf.utils import histogram


class UniformLogPrior:
    def __init__(self, label, lower_bound, upper_bound, out_of_bounds_val=-1e10):
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.out_of_bounds_val = out_of_bounds_val
        self.dist = uniform(loc=self.lower_bound, scale=self.upper_bound-self.lower_bound)

    def __call__(self, x):
        return self.dist.logpdf(x)


class GaussianLogPrior:
    def __init__(self, label, mu, sigma, lower_bound=-np.inf, upper_bound=np.inf):
        self.label = label
        self.mu = mu
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = (lower_bound - mu) / sigma
        self.b = (upper_bound - mu) / sigma
        self.dist = truncnorm(a=self.a, b=self.b, loc=self.mu, scale=self.sigma)

    def __call__(self, x):
        return self.dist.logpdf(x)


class FlatLogPrior:
    def __init__(self, label):
        self.label = label

    def __call__(self, x):
        return np.zeros_like(x)


class HistLogPrior:
    def __init__(self, label, samples, bins=25, out_of_bounds_val=1e-10):
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        self.n = samples.shape[0]
        self.label = label
        self.samples = samples
        counts, bins = histogram(samples, bins=bins, density=True)
        counts += out_of_bounds_val
        self.bins = bins
        dist = []
        for i in range(self.n):
            dist.append(rv_histogram((counts[i], bins)))
        self.dist = dist

    def __call__(self, x, shared_x=False):
        if shared_x:
            return np.vstack([self.dist[i].logpdf(x) for i in range(self.n)])
        else:
            return np.vstack([self.dist[i].logpdf(x[i]) for i in range(self.n)])


class KDELogPrior:
    def __init__(self, label, samples, xsmooth, method='Reflection', xlow=None, xhigh=None, out_of_bounds_val=1e-10, **kwargs):
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        self.n = samples.shape[0]
        self.label = label
        self.samples = samples
        self.xsmooth = xsmooth
        self.method = method
        self.xlow = xlow
        self.xhigh = xhigh
        self.out_of_bounds_val = out_of_bounds_val
        self.kwargs = kwargs
        smoothed_pdf = []
        for i in range(self.n):
            kde = bounded_1d_kde(samples[i], xlow=xlow, xhigh=xhigh, method=method, **kwargs)
            kde_eval = kde(xsmooth)
            smoothed_pdf.append(kde_eval / np.trapz(kde_eval, xsmooth))
        self.smoothed_pdf = np.vstack(smoothed_pdf)
        self.dist = interp1d(xsmooth, self.smoothed_pdf, axis=-1, bounds_error=False, fill_value=out_of_bounds_val)

    def __call__(self, x, shared_x=False):
        if shared_x:
            return np.log(self.dist(x))
        else:
            return np.log(self.dist(x).diagonal())
