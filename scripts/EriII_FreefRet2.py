"""
Eri II MDF Fitting Script: Production Run
"""
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import multiprocess as mp
import pocomc as pc
from waf.par import DefaultParSet
from waf.fitting.priors import UniformLogPrior, GaussianLogPrior, KDELogPrior
from waf.fitting.sampling import log_probability, log_likelihood, log_prior
from waf.utils import randdist
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as transforms
np.seterr(all="ignore")

# Inputs
n_walkers = 5000
n_samples = 10000
dFeH = 1e-2  # dex
dt = 1e-5  # Gyr
SFH_fn = 'exponential'
IaDTD_fn = 'powerlaw'  # -1.1 Powerlaw DTD; Maoz+ (2012)
tDminIa = 0.05  # Gyr; Citation?
r = 0.37  # Kroupa IMF after 1 Gyr
SolarFe = 0.0013  # Asplund (2009)
# SolarAlpha = 0.0056  # Alpha == O; Asplund (2009)
# yAlphaCC = 0.015  # Alpha == O; Chieffi & Limongi (2004), Limongi & Chieffi (2006), and Andrews+ (2017)
# yFeCC = 0.0015  # Match to Hayden+ (2015)
# yFeIa = 0.0013  # Match to Hayden+ (2015)
SolarAlpha = 0.0007  # Alpha == Mg; Asplund (2009)
yAlphaCC = 0.001  # Alpha == Mg;
yFeCC = 0.0006  #
yFeIa = 0.0012  #
logP_floor = -50
p0_min_logP = -100
reload_p0 = False  # Use previous p0 if it exists, skipping the costly initialization
# (set reload_p0 = False if the likelihood has changed substantially since the last run)
plotting = True
data_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_MDF.dat')
samp_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_samples.dat')
results_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/samples/EriII_FreefRet2.npz')
p0_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_FreefRet2_p0.npy')
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

# Load CaHK Abundance Posterior Samples
CaHK_samples = pd.read_csv(samp_file, index_col=0)

# Load Default Parameters
par = DefaultParSet()
par.update({
    'SFH_fn': SFH_fn,
    'IaDTD_fn': IaDTD_fn,
    'tDminIa': tDminIa,
    'r': r,
    'dt': dt,
    'SolarAlpha': SolarAlpha,
    'SolarFe': SolarFe,
    'yAlphaCC': yAlphaCC,
    'yFeCC': yFeCC,
    'yFeIa': yFeIa,
})
fine_bins = np.arange(-10, 2.0+dFeH, dFeH)

# Define Priors
CaHK_FeH_Priors = KDELogPrior('latent_FeH', CaHK_samples.values.T, fine_bins, xlow=-4)
gal_priors = dict(
    logtauSFE=UniformLogPrior('logtauSFE', 0, 4, -np.inf),
    tauSFH=GaussianLogPrior('tauSFH', 0.7, 0.3, 0.01, np.inf),
    t_trunc=GaussianLogPrior('t_trunc', 1.0, 0.5, 10*dt, 12),
    eta=UniformLogPrior('eta', 0, 1e3, -np.inf),
    fRetCC=UniformLogPrior('fRetCC', 0, 1, -np.inf),
    fRetIa=UniformLogPrior('fRetIa', 0, 1, -np.inf),
)
gal_par_names = list(gal_priors.keys())
priors = {**gal_priors, **dict(latent_FeH=CaHK_FeH_Priors)}

# Bounds for pocoMC
bounds = np.array(
    [
        [priors[key].lower_bound, priors[key].upper_bound]
        for key in gal_par_names
    ] + [
        [-4, np.nan] for i in range(CaHK_FeH_Priors.n)
    ]
)
bounds[~np.isfinite(bounds)] = np.nan

# Plot MDF
if plotting:
    plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
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
    ax1.set_xlim(-4.25, -0.75)
    ax2.set_xlim(-4.25, -0.75)
    ax2.set_ylim(0,)
    ax1.grid(False)
    ax2.grid(False)
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, prune='lower', nbins=6))
    plt.tight_layout()
    plt.savefig(fig_dir.joinpath('EriII_MDF.png'))
    plt.show()

# Initialize Walkers
if reload_p0 and p0_file.exists():
    p0 = np.load(p0_file)
else:
    # Initialize Walkers
    print(f'Generating first {int(n_walkers / 10)} walkers')
    p0_list = []
    progress_bar = tqdm(total=int(n_walkers / 10))
    while len(p0_list) < int(n_walkers / 10):
        p = np.array(
            [
                priors[key].dist.rvs() for key in gal_par_names
            ] + [
                randdist(
                    priors['latent_FeH'].xsmooth,
                    priors['latent_FeH'].smoothed_pdf[i],
                    nvals=1,
                )[0] for i in range(priors['latent_FeH'].n)
            ]
        )
        logP = log_probability(
            p,
            default_par=par,
            priors=priors,
            gal_par_names=gal_par_names,
            floor=10**logP_floor,
        )
        if logP > p0_min_logP:
            p0_list.append(p)
            progress_bar.update(1)
    # Initialize remaining walkers by bootstrapping from existing set of walkers and adding scatter
    # Not strictly initializing from the priors per se, it's substantially faster and close enough
    print(f'Generating remaining {int(n_walkers - n_walkers / 10)} walkers')
    progress_bar = tqdm(total=n_walkers, initial=len(p0_list))
    while len(p0_list) < n_walkers:
        seed_p = random.sample(p0_list, 1)[0]
        p = np.random.normal(loc=seed_p, scale=np.abs(0.05 * seed_p))
        logP = log_probability(
            p,
            default_par=par,
            priors=priors,
            gal_par_names=gal_par_names,
            floor=10**logP_floor,
        )
        if logP > p0_min_logP:
            p0_list.append(p)
            progress_bar.update(1)
    progress_bar.close()
    p0 = np.vstack(p0_list)
    np.save(p0_file, p0)
n_walkers, n_dim = p0.shape


def log_likelihood_wrapper(p, default_par, gal_par_names, floor=1e-20):
    """
    Wrapper to parse p into p_gal and p_star and pass to waf.sampling.log_likelihood
    """
    if p.ndim > 1:
        raise AttributeError('log_prior is not vectorized')
    if np.any(~np.isfinite(p)):
        return -np.inf
    p_star = p[len(gal_par_names):]
    p_gal = {par_name: p[:len(gal_par_names)][i] for i, par_name in enumerate(gal_par_names)}
    logL = log_likelihood(p_gal, p_star, default_par, floor)
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
        n_walkers,
        n_dim,
        log_likelihood=log_likelihood_wrapper,
        log_likelihood_kwargs=dict(
            default_par=par,
            gal_par_names=gal_par_names,
            floor=10**logP_floor,
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
    sampler.add_samples(n_samples - n_walkers)
# Save Results
results = sampler.results
np.savez(results_file, **results)
# Plot pocoMC Diagnostic Plots
if plotting:
    run_fig = pc.plotting.run(results)
    run_fig.savefig(fig_dir.joinpath('EriII_pocoMC_run.png'))
    trace_fig = pc.plotting.trace(
        results,
        labels=gal_par_names + [f'FeH{i:02.0f}' for i in range(n_star)],
    )
    trace_fig.savefig(fig_dir.joinpath('EriII_pocoMC_trace.png'))
