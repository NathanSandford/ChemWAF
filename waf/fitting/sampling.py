import numpy as np
from waf.models import waf2017
from waf.utils import get_PDF

def log_prior(p_gal, p_star, priors):
    logPi = np.sum([priors[key](value) for key, value in p_gal.items()])
    logPi += np.sum(priors['latent_FeH'](p_star))
    return logPi


def log_likelihood(p_gal, p_star, default_par, gal_par_names, floor=1e-10):
    default_par.update(p_gal)
    SFR, OH, FeH, OFe = waf2017(**default_par.model_kwargs)
    if ~np.all(np.isfinite(OH)) or ~np.all(np.isfinite(FeH)) or ~np.all(np.isfinite(OFe)):
        return -np.inf
    FeH_PDF, grid = get_PDF(
        FeH,
        SFR,
        grid=None,
        lower_bound=-4,
        upper_bound=None,
        boundary_dist='HalfNormal',
        boundary_width=0.35,
        floor=floor,
    )
    logL = np.sum(np.log(np.interp(p_star, grid, FeH_PDF, left=0, right=0)))
    if np.isnan(logL):
        raise RuntimeError('NaN found in logL')
    return logL


def log_probability(p, default_par, priors, gal_par_names):
    if p.ndim > 1:
        raise AttributeError('log_prior is not vectorized')
    p_star = p[len(gal_par_names):]
    p_gal = {par_name: p[:len(gal_par_names)][i] for i, par_name in enumerate(gal_par_names)}
    logPi = log_prior(p_gal, p_star, priors)
    logL = log_likelihood(p_gal, p_star, default_par, gal_par_names)
    logP = logPi + logL
    if np.isnan(logP):
        return -np.inf
    else:
        return logP


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
    except:
        n_obs_feh_bins = 1
    try:
        n_obs_oh_bins = len(obs_mdf['OH_bins']) - 1
    except:
        n_obs_oh_bins = 1
    try:
        n_obs_ofe_bins = len(obs_mdf['OFe_bins']) - 1
    except:
        n_obs_ofe_bins = 1
    SFR = np.zeros((n_draws, n_timesteps))
    OH = np.zeros((n_draws, n_timesteps))
    FeH = np.zeros((n_draws, n_timesteps))
    OFe = np.zeros((n_draws, n_timesteps))
    OH_PDF = np.zeros((n_draws,  n_grid))
    FeH_PDF = np.zeros((n_draws, n_grid))
    OFe_PDF = np.zeros((n_draws, n_grid))
    OH_MDF = np.zeros((n_draws,  n_obs_oh_bins))
    FeH_MDF = np.zeros((n_draws, n_obs_feh_bins))
    OFe_MDF = np.zeros((n_draws, n_obs_ofe_bins))
    for i in range(n_draws):
        p_star = p[i, len(gal_par_names):]
        p_gal = {par_name: p[i, :len(gal_par_names)][j] for j, par_name in enumerate(gal_par_names)}
        default_par.update(p_gal)
        sfr, oh, feh, ofe = waf2017(**default_par.model_kwargs)
        SFR[i], OH[i], FeH[i], OFe[i] = sfr, oh, feh, ofe
        FeH_PDF[i], _ = get_PDF(feh, sfr, grid=pdf_grid, lower_bound=-4, upper_bound=None, boundary_width=0.1)
        OH_PDF[i], _ = get_PDF(oh, sfr, grid=pdf_grid, lower_bound=-4, upper_bound=None, boundary_width=0.1)
        OFe_PDF[i], _ = get_PDF(ofe, sfr, grid=pdf_grid, lower_bound=-4, upper_bound=None, boundary_width=0.1)
        try:
            OH_MDF[i] = get_MDF(oh, sfr, obs_mdf['OH_bins']) * obs_mdf['OH_counts'].sum()
        except:
            OH_MDF[i] = get_MDF(oh, sfr, obs_mdf['OH_bins'])
        try:
            FeH_MDF[i] = get_MDF(feh.clip(-4,np.inf), sfr, obs_mdf['FeH_bins']) * obs_mdf['FeH_counts'].sum()
        except:
            FeH_MDF[i] = get_MDF(feh, sfr, obs_mdf['FeH_bins'])
        try:
            OFe_MDF[i] = get_MDF(ofe, sfr, obs_mdf['OFe_bins']) * obs_mdf['OFe_counts'].sum()
        except:
            OFe_MDF[i] = get_MDF(ofe, sfr, obs_mdf['OFe_bins'])
    return SFR, OH, FeH, OFe, OH_PDF, FeH_PDF, OFe_PDF, OH_MDF, FeH_MDF, OFe_MDF
