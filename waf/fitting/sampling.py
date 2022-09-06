import numpy as np
from waf.models import waf2017
from waf.utils import get_PDF, get_MDF


def log_prior(p_gal, p_star, priors):
    log_pi = np.sum([priors[key](value) for key, value in p_gal.items()])
    log_pi += np.sum(priors['latent_FeH'](p_star))
    return log_pi


def log_likelihood(p_gal, p_star, default_par, floor=1e-10):
    default_par.update(p_gal)
    sfr, oh, feh, ofe = waf2017(**default_par.model_kwargs)
    if ~np.all(np.isfinite(oh)) or ~np.all(np.isfinite(feh)) or ~np.all(np.isfinite(ofe)):
        return -np.inf
    feh_pdf, grid = get_PDF(
        feh,
        sfr,
        grid=None,
        lower_bound=-4,
        upper_bound=None,
        boundary_dist='HalfNormal',
        boundary_width=0.35,
        floor=floor,
    )
    log_like = np.sum(np.log(np.interp(p_star, grid, feh_pdf, left=0, right=0)))
    if np.isnan(log_like):
        raise RuntimeError('NaN found in log_like')
    return log_like


def log_probability(p, default_par, priors, gal_par_names, floor):
    if p.ndim > 1:
        raise AttributeError('log_prior is not vectorized')
    p_star = p[len(gal_par_names):]
    p_gal = {par_name: p[:len(gal_par_names)][i] for i, par_name in enumerate(gal_par_names)}
    log_pi = log_prior(p_gal, p_star, priors)
    log_like = log_likelihood(p_gal, p_star, default_par, floor)
    log_prob = log_pi + log_like
    if np.isnan(log_prob):
        return -np.inf
    else:
        return log_prob


def ppc(p, default_par, gal_par_names, obs_mdf, pdf_grid=None):
    if p.ndim == 1:
        p = p[np.newaxis, :]
    if pdf_grid is None:
        pdf_grid = obs_mdf['FeH_bins']
    n_draws = p.shape[0]
    n_timesteps = default_par.t.shape[0]
    n_grid = pdf_grid.shape[0]
    try:
        n_obs_feh_bins = len(obs_mdf['FeH_bins']) - 1
    except KeyError:
        n_obs_feh_bins = 1
    try:
        n_obs_oh_bins = len(obs_mdf['OH_bins']) - 1
    except KeyError:
        n_obs_oh_bins = 1
    try:
        n_obs_ofe_bins = len(obs_mdf['OFe_bins']) - 1
    except KeyError:
        n_obs_ofe_bins = 1
    sfr = np.zeros((n_draws, n_timesteps))
    oh = np.zeros((n_draws, n_timesteps))
    feh = np.zeros((n_draws, n_timesteps))
    ofe = np.zeros((n_draws, n_timesteps))
    oh_pdf = np.zeros((n_draws,  n_grid))
    feh_pdf = np.zeros((n_draws, n_grid))
    ofe_pdf = np.zeros((n_draws, n_grid))
    oh_mdf = np.zeros((n_draws,  n_obs_oh_bins))
    feh_mdf = np.zeros((n_draws, n_obs_feh_bins))
    ofe_mdf = np.zeros((n_draws, n_obs_ofe_bins))
    for i in range(n_draws):
        p_gal = {par_name: p[i, :len(gal_par_names)][j] for j, par_name in enumerate(gal_par_names)}
        default_par.update(p_gal)
        sfr[i], oh[i], feh[i], ofe[i] = waf2017(**default_par.model_kwargs)
        feh_pdf[i], _ = get_PDF(feh, sfr, grid=pdf_grid, lower_bound=-4, upper_bound=None, boundary_width=0.1)
        oh_pdf[i], _ = get_PDF(oh, sfr, grid=pdf_grid, lower_bound=-4, upper_bound=None, boundary_width=0.1)
        ofe_pdf[i], _ = get_PDF(ofe, sfr, grid=pdf_grid, lower_bound=-4, upper_bound=None, boundary_width=0.1)
        try:
            oh_mdf[i] = get_MDF(oh, sfr, obs_mdf['OH_bins']) * obs_mdf['OH_counts'].sum()
        except KeyError:
            oh_mdf[i] = get_MDF(oh, sfr, obs_mdf['OH_bins'])
        try:
            feh_mdf[i] = get_MDF(feh.clip(-4,np.inf), sfr, obs_mdf['FeH_bins']) * obs_mdf['FeH_counts'].sum()
        except KeyError:
            feh_mdf[i] = get_MDF(feh, sfr, obs_mdf['FeH_bins'])
        try:
            ofe_mdf[i] = get_MDF(ofe, sfr, obs_mdf['OFe_bins']) * obs_mdf['OFe_counts'].sum()
        except KeyError:
            ofe_mdf[i] = get_MDF(ofe, sfr, obs_mdf['OFe_bins'])
    return sfr, oh, feh, ofe, oh_pdf, feh_pdf, ofe_pdf, oh_mdf, feh_mdf, ofe_mdf
