import numpy as np
from scipy.stats import poisson
from waf.models import waf2017
from waf.utils import get_MDF


def log_prior(p, priors):
    logPi = 0
    for p_name, p_val in p.items():
        logPi += priors[p_name](p_val)
    return logPi


def log_likelihood(p, default_par, obs_mdf):
    default_par.update(p)
    SFR, OH, FeH, OFe = waf2017(**default_par.__dict__)
    OH_MDF = get_MDF(OH, SFR, obs_mdf['bins']) * obs_mdf['counts'].sum()
    FeH_MDF = get_MDF(FeH, SFR, obs_mdf['bins']) * obs_mdf['counts'].sum()
    OFe_MDF = get_MDF(OFe, SFR, obs_mdf['bins']) * obs_mdf['counts'].sum()
    logL = poisson.logpmf(obs_mdf['counts'], FeH_MDF).sum()
    return logL


def log_probability(p, default_par, obs_mdf, priors):
    logPi = log_prior(p, priors)
    logL = log_likelihood(p, default_par, obs_mdf)
    logP = logPi + logL
    if np.isnan(logP):
        return -np.inf
    else:
        return logP


def ppc(p_list, default_par, obs_mdf):
    if not isinstance(p_list, list):
        p_list = [p_list]
    SFR = np.zeros((len(p_list), len(default_par.t)))
    OH = np.zeros((len(p_list), len(default_par.t)))
    FeH = np.zeros((len(p_list), len(default_par.t)))
    OFe = np.zeros((len(p_list), len(default_par.t)))
    OH_MDF = np.zeros((len(p_list), len(obs_mdf['bins'])-1))
    FeH_MDF = np.zeros((len(p_list), len(obs_mdf['bins'])-1))
    OFe_MDF = np.zeros((len(p_list), len(obs_mdf['bins'])-1))
    for i, p in enumerate(p_list):
        default_par.update(p)
        sfr, oh, feh, ofe = waf2017(**default_par.__dict__)
        SFR[i], OH[i], FeH[i], OFe[i] = sfr, oh, feh, ofe
        OH_MDF[i] = get_MDF(oh, sfr, obs_mdf['bins']) * obs_mdf['counts'].sum()
        FeH_MDF[i] = get_MDF(feh, sfr, obs_mdf['bins']) * obs_mdf['counts'].sum()
        OFe_MDF[i] = get_MDF(ofe, sfr, obs_mdf['bins']) * obs_mdf['counts'].sum()
    return SFR, OH, FeH, OFe, OH_MDF, FeH_MDF, OFe_MDF
