import numpy as np
from waf.models import waf2017
from waf.utils import get_MDF, get_PDF, eval_PDF

def log_prior(p, priors, gal_par_names, obs=None):
    if p.ndim > 1:
        raise AttributeError('log_prior is not vectorized')
    p_gal = p[:len(gal_par_names)]
    logPi = np.sum([priors[gal_par_names[i]](p_gal[i]) for i in range(len(gal_par_names))])
    if obs is None:
        p_star = p[len(gal_par_names):]
        logPi += np.sum(priors['latent_FeH'](p_star))
    return logPi


def log_likelihood(p, default_par, gal_par_names, bins, obs=None):
    if p.ndim > 1:
        raise AttributeError('log_likelihood is not vectorized')
    p_gal = p[:len(gal_par_names)]
    p_star = p[len(gal_par_names):]
    p_gal_dict = {par_name: p_gal[i] for i, par_name in enumerate(gal_par_names)}
    default_par.update(p_gal_dict)
    SFR, OH, FeH, OFe = waf2017(**default_par.__dict__)
    FeH_PDF = get_PDF(FeH, SFR, bins)
    if obs is None:
        p_star = p[len(gal_par_names):]
    else:
        p_star = obs['FeH'].values
    logL = eval_PDF(p_star, FeH_PDF, bins)
    return logL


def log_probability(p, default_par, priors, gal_par_names, bins, obs=None):
    logPi = log_prior(p, priors, gal_par_names, obs)
    logL = log_likelihood(p, default_par, gal_par_names, bins, obs)
    logP = logPi + logL
    if np.isnan(logP):
        return -np.inf
    else:
        return logP


def ppc(p_list, default_par, bins, obs_mdf):
    if not isinstance(p_list, list):
        p_list = [p_list]
    SFR = np.zeros((len(p_list), len(default_par.t)))
    OH = np.zeros((len(p_list), len(default_par.t)))
    FeH = np.zeros((len(p_list), len(default_par.t)))
    OFe = np.zeros((len(p_list), len(default_par.t)))
    OH_MDF = np.zeros((len(p_list), len(obs_mdf['bins'])-1))
    FeH_MDF = np.zeros((len(p_list), len(obs_mdf['bins'])-1))
    OFe_MDF = np.zeros((len(p_list), len(obs_mdf['bins'])-1))
    OH_PDF = np.zeros((len(p_list), len(bins)-1))
    FeH_PDF = np.zeros((len(p_list), len(bins)-1))
    OFe_PDF = np.zeros((len(p_list), len(bins)-1))
    for i, p in enumerate(p_list):
        default_par.update(p)
        sfr, oh, feh, ofe = waf2017(**default_par.__dict__)
        SFR[i], OH[i], FeH[i], OFe[i] = sfr, oh, feh, ofe
        OH_MDF[i] = get_MDF(oh, sfr, obs_mdf['bins']) * obs_mdf['counts'].sum()
        FeH_MDF[i] = get_MDF(feh, sfr, obs_mdf['bins']) * obs_mdf['counts'].sum()
        OFe_MDF[i] = get_MDF(ofe, sfr, obs_mdf['bins']) * obs_mdf['counts'].sum()
        OH_PDF[i] = get_PDF(oh, sfr, bins)
        FeH_PDF[i] = get_PDF(feh, sfr, bins)
        OFe_PDF[i] = get_PDF(ofe, sfr, bins)
    return SFR, OH, FeH, OFe, OH_MDF, FeH_MDF, OFe_MDF, OH_PDF, FeH_PDF, OFe_PDF
