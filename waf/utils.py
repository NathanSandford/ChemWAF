import numpy as np

def get_MDF(
    x: np.ndarray,
    sfr: np.ndarray,
    bins: np.ndarray,
):
    x_MDF, _ = np.histogram(x, weights=sfr, bins=bins)
    x_MDF /= x_MDF.sum()
    return x_MDF


def get_PDF(
        x: np.ndarray,
        sfr: np.ndarray,
        bins: np.ndarray,
):
    x_PDF = get_MDF(x, sfr, bins)
    bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2
    x_PDF += 1e-10
    x_PDF /= np.sum(np.diff(bins) * x_PDF)
    return x_PDF


def eval_PDF(
        obs: np.ndarray,
        pdf: np.ndarray,
        bins: np.ndarray,
):
    bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2
    logL = np.sum(np.log(np.interp(obs, bin_centers, pdf)))
    return logL


def randdist(x, pdf, nvals):
    """Produce nvals random samples from pdf(x), assuming constant spacing in x."""
    # get cumulative distribution from 0 to 1
    if np.diff(x).max() -  np.diff(x).min() > 1e-10:
        raise RuntimeError('x must have constant spacing')
    cumpdf = np.cumsum(pdf)
    cumpdf *= 1/cumpdf[-1]
    # input random values
    randv = np.random.uniform(size=nvals)
    # find where random values would go
    idx1 = np.searchsorted(cumpdf, randv)
    # get previous value, avoiding division by zero below
    idx0 = np.where(idx1==0, 0, idx1-1)
    idx1[idx0==0] = 1
    # do linear interpolation in x
    frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
    randdist = x[idx0]*(1-frac1) + x[idx1]*frac1
    return randdist
