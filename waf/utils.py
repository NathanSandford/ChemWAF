from typing import Optional
from copy import deepcopy
import numpy as np
from scipy.stats import uniform, halfnorm

def get_MDF(
    x: np.ndarray,
    sfr: np.ndarray,
    bins: np.ndarray,
):
    x_MDF, _ = np.histogram(x, weights=sfr, bins=bins)
    x_MDF /= x_MDF.sum()
    return x_MDF


def get_PDF(
    val: np.array,
    weights: np.array,
    grid: Optional[np.array] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    boundary_dist: str = 'HalfNormal',
    boundary_width: Optional[float] = None,
    floor: float = 0,
):
    '''

    :param val:
    :param weights:
    :param grid:
    :param lower_bound:
    :param upper_bound:
    :param boundary_width:
    :param floor:
    :return:
    '''
    if grid is None:
        # Set grid for interpolation in val-space if none is given
        grid = np.linspace(val.min(), val.max(), 1001)
    if np.all(np.diff(val) > 0):
        pdf_method = 'strictly monotonically increasing'
        pdf_raw = weights[1:] / np.diff(val)
        pdf_grid = np.interp(grid, val[1:], pdf_raw, left=floor, right=floor)
        pdf_grid /= np.trapz(pdf_grid, grid)
    elif np.all(np.diff(val) < 0):
        pdf_method = 'strictly monotonically decreasing'
        pdf_raw = weights[1:][::-1] / np.diff(val[::-1])
        pdf_grid = np.interp(grid, val[1:][::-1], pdf_raw, left=floor, right=floor)
        pdf_grid /= np.trapz(pdf_grid, grid)
    elif np.all(np.diff(val) >= 0):
        pdf_method = 'monotonically increasing (w/ zeros)'
        # Fall back on histogram method
        pdf_raw, _ = np.histogram(val, weights=weights, bins=grid, density=True)
        bin_centers = np.convolve(grid, np.ones(2), 'valid') / 2
        pdf_grid = np.interp(grid, bin_centers, pdf_raw, left=floor, right=floor)
        pdf_grid /= np.trapz(pdf_grid, grid)
    else:
        pdf_method = 'non-monotonic'
        idx_inc = np.diff(val) > 0
        idx_inc = np.concatenate([[np.diff(val)[0] > 0], idx_inc])
        idx_dec = np.diff(val) < 0
        idx_dec = np.concatenate([[np.diff(val)[0] < 0], idx_dec])
        val_inc = val[idx_inc]
        val_dec = val[idx_dec]
        # Handle increasing portion
        pdf_raw_inc = weights[idx_inc][1:] / np.diff(val_inc)
        pdf_interp_inc = np.interp(grid, val_inc[1:], pdf_raw_inc, left=floor, right=floor)
        # Handle decreasing portion (in reverse)
        pdf_raw_dec = weights[idx_dec][:-1][::-1] / np.diff(val_dec[::-1])
        pdf_interp_dec = np.interp(grid, val_dec[:-1][::-1], pdf_raw_dec, left=floor, right=floor)
        # Combine increasing and decreasing portion
        pdf_grid = pdf_interp_inc + pdf_interp_dec
        pdf_grid /= np.trapz(pdf_grid, grid)
    if np.any(pdf_grid < 0):
        raise RuntimeError('Negative PDF value detected')
    # Censor the PDF is a lower/upper-limit is provided
    censored_pdf = censor_distribution(pdf_grid, grid, lower_bound, upper_bound, boundary_dist, boundary_width)
    return censored_pdf, grid


def censor_distribution(
    dist: np.array,
    x: np.array,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    boundary_dist: str = 'HalfNormal',
    boundary_width: Optional[float] = None,
):
    """

    :param dist:
    :param x:
    :param lower_bound:
    :param upper_bound:
    :param boundary_dist:
    :param boundary_width:
    :return:
    """
    censored_dist = deepcopy(dist)
    # Get boundary width in pixels (bins)
    if boundary_width is None:
        boundary_width_pixels = 1
    else:
        boundary_width_pixels = int(boundary_width / np.diff(x)[0])
    if (
        (lower_bound is not None)  # If there is a lower_bound provided
        and (x[0] < lower_bound)  # If there are any values to censor
        and (x[-1] > lower_bound)  # If the whole distribution is not censored
    ):
        # First non-censored index
        lower_bound_idx1 = np.argmax(x >= lower_bound)
        # End of boundary region
        lower_bound_idx2 = lower_bound_idx1 + boundary_width_pixels
        # Censor all values below lower_bound
        censored_dist[x < lower_bound] = 0
        # Add truncated mass to boundary region
        truncated_mass = 1 - np.trapz(censored_dist, x)
        if truncated_mass > 0:
            if boundary_dist.lower() == 'delta':
                censored_dist[lower_bound_idx1] += truncated_mass / np.diff(x)[0]
            elif boundary_dist.lower() in ['norm', 'normal', 'halfnorm', 'halfnormal']:
                censored_dist += truncated_mass * halfnorm.pdf(x, loc=lower_bound, scale=boundary_width)
            elif boundary_dist.lower() in ['uniform', 'flat']:
                censored_dist += truncated_mass * uniform.pdf(x, loc=lower_bound, scale=boundary_width)
            else:
                raise NotImplementedError(f'{boundary_dist} is not implemented')
            #censored_dist[lower_bound_idx1:lower_bound_idx2] \
            #    += truncated_mass / (np.diff(x)[0] * boundary_width_pixels)
    if (
        (upper_bound is not None)  # If there is a upper_bound provided
        and (x[-1] > upper_bound)  # If there are any values to censor
        and (x[0] < upper_bound)  # If the whole distribution is not censored
    ):
        # Last non-censored index
        upper_bound_idx1 = np.argmax(x >= upper_bound)
        # Beginning of boundary region
        upper_bound_idx2 = max(0, upper_bound_idx1 - boundary_width_pixels)
        # Censor all values above upper_bound
        censored_dist[x > upper_bound] = 0
        # Add truncated mass to boundary region
        truncated_mass = 1 - np.trapz(censored_dist, x)
        if truncated_mass > 0:
            if boundary_dist.lower() == 'delta':
                censored_dist[upper_bound_idx1] += truncated_mass / np.diff(x)[0]
            elif boundary_dist.lower() in ['halfnorm', 'halfnormal']:
                censored_dist += truncated_mass * halfnorm.pdf(-x, loc=-upper_bound, scale=boundary_width)
            elif boundary_dist.lower() in ['uniform', 'flat']:
                censored_dist += truncated_mass * uniform.pdf(-x, loc=-upper_bound, scale=boundary_width)
            else:
                raise NotImplementedError(f'{boundary_dist} is not implemented')
            #censored_dist[upper_bound_idx2+1:upper_bound_idx1+1] \
            #    += truncated_mass / (np.diff(x)[0] * boundary_width_pixels)
    censored_dist /= np.trapz(censored_dist, x)
    if np.any(censored_dist < 0):
        raise RuntimeError('Negative PDF value detected')
    return censored_dist


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


def histogram(data, bins, density=False):
    """
    Adapted from https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
    """
    R = [data.min(), data.max()]
    N = data.shape[-1]
    if isinstance(bins, int):
        bins = np.linspace(R[0], R[1], bins+1)
    elif bins.lower() == 'auto':
        bins = np.histogram_bin_edges(data.flatten(), bins='auto')
    bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2
    data2D = data.reshape(-1, N)
    idx = np.searchsorted(bins, data2D, 'right')-1
    bad_mask = (idx==-1) | (idx==len(bins))
    scaled_idx = len(bins) * np.arange(data2D.shape[0])[:, None] + idx
    limit = len(bins) * data2D.shape[0]
    scaled_idx[bad_mask] = limit
    counts = np.bincount(scaled_idx.ravel(), minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (len(bins),)
    if density:
        counts = counts[:, :-1] / np.trapz(counts[:, :-1], bin_centers, axis=1)[:, np.newaxis]
    return counts, bins
