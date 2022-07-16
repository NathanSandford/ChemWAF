from pathlib import Path
import numpy as np
import pandas as pd
import multiprocess as mp
import pocomc as pc
from waf.par import DefaultParSet
from waf.fitting.priors import UniformLogPrior, GaussianLogPrior
from waf.fitting.pocomc import log_prior_0bins as log_prior
from waf.fitting.pocomc import log_likelihood_0bins as log_likelihood
from waf.fitting.pocomc import log_probability_0bins as log_probability
from waf.fitting.pocomc import ppc
import matplotlib as mpl
import matplotlib.pyplot as plt
from corner import corner
np.seterr(all="ignore");

nwalkers = 5000
use_obs_errs = False
plot_p0 = True
plot_pocomc = True
plot_corner = True
plot_ppc = True
data_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_MDF.dat')
output_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/SFE_SFH_eta_0bins.npz')

# Load Observed Data
eri_ii = pd.read_csv(data_file, index_col=0)
obs_bins = np.linspace(-10, 2.0, 49)
counts, obs_bins = np.histogram(eri_ii['FeH'], bins=obs_bins, density=False)
eri_ii_mdf = {
    'counts': counts,
    'bins': obs_bins,
}
n_obj = eri_ii.shape[0]
# Load Default Parameters
par = DefaultParSet()
par.t = np.arange(0.0001, 1.0001, 0.0001)
mod_bins = np.linspace(-10, 2.0, 500)
# Define Priors
gal_priors = dict(
    logtauSFE=UniformLogPrior('logtauSFE', 0, 4, -np.inf),
    tauSFH=GaussianLogPrior('tauSFH', 0.7, 0.2, 0, 1e2),
    eta=UniformLogPrior('eta', 0, 1e3, -np.inf),
)
gal_par_names = list(gal_priors.keys())
bounds = np.array([
    [0, 4],  # logtauSFE
    [0, 1e2],  # tauSFH
    [0, 1e3],  # eta
])
if use_obs_errs:
    bounds = np.concatenate([bounds, [[np.nan, np.nan] for i in range(n_obj)]])
    star_priors = dict(latent_FeH=GaussianLogPrior('latent_FeH', eri_ii['FeH'], eri_ii['dFeH']))
    priors = {**gal_priors, **star_priors}
else:
    priors = gal_priors
# Initialize Walkers
p0_list = []
while len(p0_list) < nwalkers:
    p = np.array(
    [
        2.0 + 1e-1 * np.random.randn(),  # logtauSFE
        0.70 + 1e-1 * np.random.randn(),  # tauSFH
        100 + 1e+1 * np.random.randn(),  # eta
    ])
    if use_obs_errs:
        p = np.concatenate([p, [eri_ii['FeH'][i] + eri_ii['dFeH'][i] * np.random.randn() for i in range(n_obj)]])
    if np.isfinite(
        log_probability(
            p,
            par,
            priors,
            gal_par_names,
            mod_bins,
            obs=None if use_obs_errs else eri_ii,
        )
    ):
        p0_list.append(p)
p0 = np.vstack(p0_list)
nwalkers, ndim = p0.shape
if plot_p0:
    # Plot Observed MDF & MDF of Initial Walkers
    p0_dict_list = [{par_name: p0[i,j] for j, par_name in enumerate(gal_par_names)} for i in range(nwalkers)]
    SFR, OH, FeH, OFe, OH_MDF, FeH_MDF, OFe_MDF, OH_PDF, FeH_PDF, OFe_PDF = ppc(p0_dict_list, par, mod_bins, eri_ii_mdf)
    plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    ax.stairs(eri_ii_mdf['counts'], eri_ii_mdf['bins'], color='k', lw=3, label='Fu+ 2022')
    ax.scatter(eri_ii['FeH'], np.ones_like(eri_ii['FeH']), marker='|', c='k', s=100)
    ax.errorbar(0, 1, xerr=np.median(eri_ii['dFeH']), fmt='ok', capsize=10)
    ax.text(0, 1.75, 'Median Error', fontsize=24, ha='center')
    ax.stairs(
        np.percentile(FeH_MDF, 50, axis=0),
        eri_ii_mdf['bins'],
        color='r', lw=3, label='Median Prediction',
    )
    ax.stairs(
        np.percentile(FeH_MDF, 97.5, axis=0),
        eri_ii_mdf['bins'],
        baseline=np.percentile(FeH_MDF, 2.5, axis=0),
        fill=True, alpha=0.2, color='r', label='95% CI',
    )
    ax.legend(fontsize=24)
    ax.set_xlabel('[Fe/H]', fontsize=36)
    ax.set_ylabel(r'$N_\mathrm{star}$', fontsize=36)
    ax.tick_params('x', labelsize=24)
    ax.tick_params('y', labelsize=24)
    ax.set_xlim(-6,0.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
# Run PMC Sampling
with mp.Pool(mp.cpu_count()) as pool:
    # Sampler initialisation
    sampler = pc.Sampler(
        nwalkers,
        ndim,
        log_likelihood=log_likelihood,
        log_likelihood_kwargs=dict(
            default_par=par,
            gal_par_names=gal_par_names,
            bins=mod_bins,
            obs=None if use_obs_errs else eri_ii,
        ),
        log_prior=log_prior,
        log_prior_kwargs=dict(
            priors=priors,
            gal_par_names=gal_par_names,
            obs=None if use_obs_errs else eri_ii,
        ),
        bounds=bounds,
        pool=pool,
    )
    # Run sampler
    sampler.run(p0)
    # Add extra samples
    sampler.add_samples(9000)
# Save Results
results = sampler.results
np.savez(output_file, **results)