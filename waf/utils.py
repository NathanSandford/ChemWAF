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
