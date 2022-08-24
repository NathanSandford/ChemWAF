"""
Eri II MDF Fitting Script: Production Run
"""
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocess as mp
import pocomc as pc
from waf.par import DefaultParSet
from waf.fitting.priors import UniformLogPrior, GaussianLogPrior, KDELogPrior
from waf.fitting.sampling import log_likelihood, log_prior
from waf.utils import randdist
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as transforms

# Inputs
nwalkers = 5000
dt = 0.001  # Gyr
t_trunc = 1.0  # Gyr
p0_min_logP = -150  # -np.inf
plotting = True
data_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_MDF.dat')
samp_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_samples.dat')
fig_dir = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/figures')

# Matplotlib defaults
mpl.rc('axes', grid=True, lw=2)
mpl.rc('ytick', direction='in', labelsize=10)
mpl.rc('ytick.major', size=5, width=1)
mpl.rc('xtick', direction='in', labelsize=10)
mpl.rc('xtick.major', size=5, width=1)
mpl.rc('ytick', direction='in', labelsize=10)
mpl.rc('ytick.major', size=5, width=1)
mpl.rc('grid', alpha=0.75, lw=1)
mpl.rc('legend', edgecolor='k', framealpha=1, fancybox=False)
mpl.rc('figure', dpi=100)

# Load Binned CaHK Abundances
eri_ii = pd.read_csv(data_file, index_col=0)
n_star = len(eri_ii)
obs_bins = np.arange(-7.5, 0.56, 0.35)  # Matching Fu+2022
counts, obs_bins = np.histogram(eri_ii['FeH'], bins=obs_bins, density=False)
eri_ii_mdf = {
    'FeH_counts': counts,
    'FeH_bins': obs_bins,
}

# Load CaHK Abundance Poterior Samples
samples = pd.read_csv(samp_file, index_col=0)

# Load Default Parameters
par = DefaultParSet()
par.t = np.arange(dt, t_trunc+dt, dt)
fine_bins = np.linspace(-10, 2.0, 1201)  # d[Fe/H] = 0.01 dex

# Define Priors
CaHK_FeH_Priors = KDELogPrior('latent_FeH', samples.values.T, fine_bins, xlow=-4)
gal_priors = dict(
    logtauSFE=UniformLogPrior('logtauSFE', 0, 4, -np.inf),
    tauSFH=GaussianLogPrior('tauSFH', 0.7, 0.3, 0, np.inf),
    eta=UniformLogPrior('eta', 0, 1e3, -np.inf),
    fRetCC=UniformLogPrior('fRetCC', 0, 1, -np.inf),
    fRetIa=UniformLogPrior('fRetIa', 0, 1, -np.inf),
)
gal_par_names = list(gal_priors.keys())
priors = {**gal_priors, **dict(latent_FeH=CaHK_FeH_Priors)}
bounds = np.array(
    [
        [priors[key].lower_bound, priors[key].upper_bound]
        for key in gal_par_names
    ] + [
        [np.nan, np.nan] for i in range(CaHK_FeH_Priors.n)
    ]
)

# Plot MDF
if plotting:
    plt.figure(figsize=(16,12))
    gs = GridSpec(2, 1, height_ratios=[1,1])
    gs.update(hspace=0.0, wspace=0.0)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    ax1.set_title('Eri II MDF (Fu+ 2022)', fontsize=48)
    ax1.text(
        0.98, 0.90, f'$N = {n_star}$', fontsize=48,
        transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top',
    )
    ax1.stairs(eri_ii_mdf['FeH_counts'], eri_ii_mdf['FeH_bins'], color='k', lw=3)
    ax1.scatter(
        eri_ii['FeH'], 0.05*np.ones_like(eri_ii['FeH']),
        marker='|', c='k', s=100,
        transform=transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    )
    CaHK_FeH_prior_vals = np.exp(CaHK_FeH_Priors(fine_bins, shared_x=True))
    for i in range(n_star):
        ax2.plot(fine_bins, CaHK_FeH_prior_vals[i], c='k', alpha=0.5)
    ax1.axvline(-4.0, c='r', lw=3, ls='--')
    ax2.axvline(-4.0, c='r', lw=3, ls='--')
    ax2.text(
        -4.08, 0.98, 'Model Grid Boundary', c='r', fontsize=36,
        horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax2.get_xaxis_transform()
        )
    ax1.set_xlabel('[Fe/H]', fontsize=0)
    ax2.set_xlabel('[Fe/H]', fontsize=48)
    ax1.set_ylabel(r'$N_\mathrm{star}$', fontsize=48)
    ax2.set_ylabel('CaHK\nPosteriors', fontsize=48)
    ax1.tick_params('x', labelsize=0)
    ax1.tick_params('y', labelsize=36)
    ax2.tick_params('x', labelsize=36)
    ax2.tick_params('y', labelsize=0, length=0)
    ax1.set_xlim(-4.25,-0.75)
    ax2.set_xlim(-4.25,-0.75)
    ax2.set_ylim(0,)
    ax1.grid(False)
    ax2.grid(False)
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, prune='lower', nbins=6))
    plt.tight_layout()
    plt.savefig(fig_dir.joinpath('EriII_MDF.png'))
    plt.show()

# Initialize Walkers
p0_list = []
pbar = tqdm(total=nwalkers)
while len(p0_list) < nwalkers:
    p = np.array(
        [
            priors[key].dist.rvs() for key in gal_par_names
        ] + [
            #priors['latent_FeH'].dist[0].rvs() for i in range(CaHK_FeH_Priors.n)
            randdist(
                priors['latent_FeH'].xsmooth,
                priors['latent_FeH'].smoothed_pdf[i],
                nvals=1,
            )[0] for i in range(priors['latent_FeH'].n)
        ]
    )
    if logP > p0_min_logP:
        p0_list.append(p)
        pbar.update(1)
pbar.close()
p0 = np.vstack(p0_list)
nwalkers, ndim = p0.shape


def log_likelihood_wrapper(p, default_par, gal_par_names):
    """
    Wrapper to parse p into p_gal and p_star and pass to waf.sampling.log_likelihood
    """
    if p.ndim > 1:
        raise AttributeError('log_prior is not vectorized')
    p_star = p[len(gal_par_names):]
    p_gal = {par_name: p[:len(gal_par_names)][i] for i, par_name in enumerate(gal_par_names)}
    logL = log_likelihood(p_gal, p_star, default_par, gal_par_names)
    return logL


def log_prior_wrapper(p, priors, gal_par_names):
    """
    Wrapper to parse p into p_gal and p_star and pass to waf.sampling.log_prior
    """
    if p.ndim > 1:
        raise AttributeError('log_prior is not vectorized')
    p_star = p[len(gal_par_names):]
    p_gal = {par_name: p[:len(gal_par_names)][i] for i, par_name in enumerate(gal_par_names)}
    logPi = log_prior(p_gal, p_star, priors)
    return logPi


# Run PMC Sampling
with mp.Pool(mp.cpu_count()) as pool:
    # Sampler initialisation
    sampler = pc.Sampler(
        nwalkers,
        ndim,
        log_likelihood=log_likelihood_wrapper,
        log_likelihood_kwargs=dict(
            default_par=par,
            gal_par_names=gal_par_names,
        ),
        log_prior=log_prior_wrapper,
        log_prior_kwargs=dict(
            priors=priors,
            gal_par_names=gal_par_names,
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
# Plot pocoMC Diagnostic Plots
if plotting:
    run_fig = pc.plotting.run(results)
    run_fig.savefig(fig_dir.joinpath('EriII_pocoMC_run.png'))
    trace_fig = pc.plotting.trace(
        results,
        labels=gal_par_names + [f'FeH{i:02.0f}' for i in range(n_star)],
    )
    trace_fig.savefig(fig_dir.joinpath('EriII_pocoMC_trace.png'))
