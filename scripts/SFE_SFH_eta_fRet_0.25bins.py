from pathlib import Path
import numpy as np
import pandas as pd
import multiprocess as mp
import emcee
from waf.par import DefaultParSet
from waf.fitting.priors import UniformLogPrior, GaussianLogPrior
from waf.fitting.emcee import log_probability, ppc
import matplotlib as mpl
import matplotlib.pyplot as plt
from corner import corner
np.seterr(all="ignore");

nwalkers = 512
plot_p0 = False
plot_chains = False
plot_corner = False
plot_ppc = False
data_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_MDF.dat')
output_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_samples.h5')

# Load Observed Data
eri_ii = pd.read_csv(data_file, index_col=0)
obs_bins = np.linspace(-10, 2.0, 49)
counts, obs_bins = np.histogram(eri_ii['FeH'], bins=obs_bins, density=False)
eri_ii_mdf = {
    'counts': counts,
    'bins': obs_bins,
}
# Load Default Parameters
par = DefaultParSet()
par.t = np.arange(0.001, 1.001, 0.001)
# Define Priors
priors = dict(
    logtauSFE=UniformLogPrior('logtauSFE', 0, 4, -np.inf),
    tauSFH=GaussianLogPrior('tauSFH', 0.7, 0.2, 0, 1e2),
    eta=UniformLogPrior('eta', 0, 1e3, -np.inf),
    fRet=GaussianLogPrior('fRet', 1.0, 0.3, 0, 1),
)
# Initialize Walkers
p0_dict = []
while len(p0_dict) < nwalkers:
    p = dict(
        logtauSFE=2.0 + 1e-1 * np.random.randn(),
        tauSFH=0.70 + 1e-1 * np.random.randn(),
        eta=100 + 1e+1 * np.random.randn(),
        fRet=0.8 + 1e-1 * np.random.randn(),
    )
    if np.isfinite(log_probability(p, par, eri_ii_mdf, priors)):
        p0_dict.append(p)
labels = list(p0_dict[0].keys())
ndim = len(labels)
p0 = np.vstack([list(p0_dict[i].values()) for i in range(nwalkers)])
SFR, OH, FeH, OFe, OH_MDF, FeH_MDF, OFe_MDF = ppc(p0_dict, par, eri_ii_mdf)
if plot_p0:
    # Plot Observed MDF & MDF of Initial Walkers
    plt.figure(figsize=(16, 8))
    ax = plt.subplot(111)
    ax.stairs(eri_ii_mdf['counts'], eri_ii_mdf['bins'], color='k', lw=3, label='Fu+ 2022')
    ax.scatter(eri_ii['FeH'], np.ones_like(eri_ii['FeH']), marker='|', c='k', s=100)
    ax.errorbar(0, 1, xerr=np.median(eri_ii['dFeH']), fmt='ok', capsize=10)
    ax.text(0, 1.75, 'Median Error', fontsize=24, ha='center')
    ax.stairs(
        np.percentile(FeH_MDF, 50, axis=0),
        eri_ii_mdf['bins'],
        color='r', lw=3, label='Median p0 Prediction',
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
    ax.set_xlim(-6, 0.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

# Run MCMC Sampling
pool = mp.Pool(mp.cpu_count())
backend = emcee.backends.HDFBackend(output_file, name=f"SFE_SFH_eta_fRet_0.25bins")
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability,
    kwargs=dict(
        default_par=par,
        obs_mdf=eri_ii_mdf,
        priors=priors,
    ),
    parameter_names=labels,
    vectorize=False,
    pool=pool,
    backend=backend,
)
max_steps = 10000
old_tau = np.inf
for _ in sampler.sample(p0, iterations=max_steps, progress=False, store=True):
    if sampler.iteration % 100:
        continue
    tau = sampler.get_autocorr_time(
        discard=int(np.max(old_tau)) if np.all(np.isfinite(old_tau)) else 0,
        tol=0
    )
    print(
        f"{sampler.iteration}: " +
        f"Tau = {np.max(tau):.0f}, " +
        f"t/50Tau = {sampler.iteration / (50 * np.max(tau)):.2f}" +
        f"\ndTau/Tau = {np.max(np.abs(old_tau - tau) / tau):.3f}, " +
        f"mean acceptance fraction = {sampler.acceptance_fraction.mean():.2f}"
    )
    # Check convergence
    converged = np.all(tau * 50 < sampler.iteration)
    converged &= np.all((tau - old_tau) / tau < 0.01)
    old_tau = tau
    if converged:
        break
    if sampler.iteration == max_steps:
        print(
            'Maximum # of steps reached before convergence'
        )

if plot_chains:
    # Plot Chains
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.show()
    # Calculate Autocorrelation Time
    tau = sampler.get_autocorr_time()
    print(tau)
if plot_corner:
    # Plot Corner
    flat_samples = sampler.get_chain(discard=int(5*np.max(tau)), thin=int(np.max(tau)/2), flat=True)
    fig = plt.figure(figsize=(10,10))
    fig = corner(
        flat_samples, labels=labels, quantiles=(0.16, 0.5, 0.84), fig=fig, show_titles=True, label_kwargs=dict(fontsize=20), hist_kwargs=dict(density=True)
    );
    # Overplot Priors
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i, label in enumerate(labels):
        ax = axes[i, i]
        x = np.linspace(0.75*flat_samples[:, i].min(), 1.25*flat_samples[:, i].max(), 10000)
        ax.plot(x, np.exp(priors[label](x)), color="g", label='prior' if i == 0 else '')
        if i == 0:
            ax.legend()
    plt.show()
if plot_ppc:
    ppc_samples_dict = [{label: flat_samples[i,j] for j, label in enumerate(labels)} for i in range(samples.shape[0])]
    SFR, OH, FeH, OFe, OH_MDF, FeH_MDF, OFe_MDF, OH_PDF, FeH_PDF, OFe_PDF = ppc(ppc_samples_dict, par, obs_bins, eri_ii_mdf)
    # Plot Observed MDF & MDF of Initial Walkers
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
    # Plot [O/H] PPC
    plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    ax.stairs(
        np.percentile(OH_MDF, 50, axis=0),
        eri_ii_mdf['bins'],
        color='r', lw=3, label='Median Prediction',
    )
    ax.stairs(
        np.percentile(OH_MDF, 97.5, axis=0),
        eri_ii_mdf['bins'],
        baseline=np.percentile(OH_MDF, 2.5, axis=0),
        fill=True, alpha=0.2, color='r', label='95% CI',
    )
    ax.legend(fontsize=24)
    ax.set_xlabel('[O/H]', fontsize=36)
    ax.set_ylabel(r'$N_\mathrm{star}$', fontsize=36)
    ax.tick_params('x', labelsize=24)
    ax.tick_params('y', labelsize=24)
    ax.set_xlim(-6,0.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
    # Plot [O/Fe] PPC
    plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    ax.stairs(
        np.percentile(OFe_MDF, 50, axis=0),
        eri_ii_mdf['bins'],
        color='r', lw=3, label='Median Prediction',
    )
    ax.stairs(
        np.percentile(OFe_MDF, 97.5, axis=0),
        eri_ii_mdf['bins'],
        baseline=np.percentile(OFe_MDF, 2.5, axis=0),
        fill=True, alpha=0.2, color='r', label='95% CI',
    )
    ax.legend(fontsize=24)
    ax.set_xlabel('[O/Fe]', fontsize=36)
    ax.set_ylabel(r'$N_\mathrm{star}$', fontsize=36)
    ax.tick_params('x', labelsize=24)
    ax.tick_params('y', labelsize=24)
    ax.set_xlim(-2,2)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
    # Plot Time Evolution PPC
    plt.figure(figsize=(16,8))
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(par.t[::100],  np.percentile(FeH, 50, axis=0)[::100], color='r', lw=3, label='[Fe/H]')
    ax1.fill_between(par.t[::100], np.percentile(FeH, 2.5, axis=0)[::100], np.percentile(FeH, 97.5, axis=0)[::100], alpha=0.2, color='r',)
    ax1.plot(par.t[::100],  np.percentile(OH, 50, axis=0)[::100], color='b', lw=3, label='[O/H]')
    ax1.fill_between(par.t[::100], np.percentile(OH, 2.5, axis=0)[::100], np.percentile(OH, 97.5, axis=0)[::100], alpha=0.2, color='b',)
    ax1.plot(par.t[::100],  np.percentile(OFe, 50, axis=0)[::100], color='purple', lw=3, label='[O/Fe]')
    ax1.fill_between(par.t[::100], np.percentile(OFe, 2.5, axis=0)[::100], np.percentile(OFe, 97.5, axis=0)[::100], alpha=0.2, color='purple',)
    ax2.plot(par.t[::100],  np.percentile(SFR, 50, axis=0)[::100], color='grey', lw=3, label='Fractional SFR')
    ax2.fill_between(par.t[::100], np.percentile(SFR, 2.5, axis=0)[::100], np.percentile(SFR, 97.5, axis=0)[::100], alpha=0.2, color='grey',)
    ax1.legend(fontsize=24, loc='upper left')
    ax2.legend(fontsize=24, loc='lower left')
    ax1.set_xlabel('[X/Y]', fontsize=36)
    ax2.set_ylabel(r'dSFR/dt', fontsize=36)
    ax1.tick_params('x', labelsize=24)
    ax1.tick_params('y', labelsize=24)
    plt.tight_layout()
    plt.show()
    # Plot [O/Fe] vs. [Fe/H] PPC
    OFe_interp = np.zeros((flat_samples.shape[0], obs_bins.shape[0]))
    for i in range(flat_samples.shape[0]):
        OFe_interp[i] = np.interp(obs_bins, FeH[i], OFe[i], left=np.nan, right=np.nan)
    plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    ax.plot(obs_bins, np.percentile(OFe_interp, 50, axis=0), color='r', lw=3)
    ax.fill_between(
        obs_bins[obs_bins < np.percentile(FeH.max(axis=1), 97.5)],
        np.nanpercentile(OFe_interp, 2.5, axis=0)[obs_bins < np.percentile(FeH.max(axis=1), 97.5)],
        np.nanpercentile(OFe_interp, 97.5, axis=0)[obs_bins < np.percentile(FeH.max(axis=1), 97.5)],
        alpha=0.2, color='r',)
    ax.set_ylabel('[O/Fe]', fontsize=36)
    ax.set_xlabel('[Fe/H]', fontsize=36)
    ax.set_xlim(-4.0, -1.0)
    ax.set_ylim(0.05, 0.35)
    ax.tick_params('x', labelsize=24)
    ax.tick_params('y', labelsize=24)
    plt.show()
