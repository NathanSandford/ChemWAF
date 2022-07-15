import numpy as np
import pandas as pd
import multiprocess as mp
import pocomc as pc
from waf.par import DefaultParSet
from waf.fitting.priors import UniformLogPrior, GaussianLogPrior
from waf.fitting.emcee import log_probability, ppc
import matplotlib as mpl
import matplotlib.pyplot as plt
from corner import corner
np.seterr(all="ignore");

nwalkers = 5000
use_obs_errs = True
plot_p0 = False
plot_pocomc = False
plot_corner = False
plot_ppc = False

# Load Default Parameters
par = DefaultParSet()
par.t = np.arange(0.0001, 1.0001, 0.0001)
mod_bins = np.linspace(-10, 2.0, 500)
# Define Priors
gal_priors = dict(
    logtauSFE=UniformLogPrior('logtauSFE', 0, 4, -np.inf),
    tauSFH=GaussianLogPrior('tauSFH', 0.7, 0.2, 0, 1e2),
    eta=UniformLogPrior('eta', 0, 1e3, -np.inf),
    fRetCC=GaussianLogPrior('fRetCC', 1.0, 0.3, 0, 1),
    fRetIa=GaussianLogPrior('fRetIa', 1.0, 0.3, 0, 1),
)
gal_par_names = list(gal_priors.keys())
bounds = np.array([
    [0, 4],  # logtauSFE
    [0, 1e2],  # tauSFH
    [0, 1e3],  # eta
    [0, 1],  # fRetCC
    [0, 1],  # fRetIa
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
        0.8 + 1e-1 * np.random.randn(),  # fRetCC
        0.8 + 1e-1 * np.random.randn(),  # fRetIa
    ])
    if use_obs_errs:
        p = np.concatenate([p, [eri_ii['FeH'][i] + eri_ii['dFeH'][i] * np.random.randn() for i in range(n_obj)]])
    if np.isfinite(
        log_probability(
            p,
            par,
            priors,
            gal_par_names,
            bins,
            obs=None if use_obs_errs else eri_ii,
        )
    ):
        p0_list.append(p)
p0 = np.vstack(p0_list)
nwalkers, ndim = p0.shape
if plot_p0:
    # Plot Observed MDF & MDF of Initial Walkers
    p0_dict_list = [{par_name: p0[i,j] for j, par_name in enumerate(gal_par_names)} for i in range(nwalkers)]
    SFR, OH, FeH, OFe, OH_MDF, FeH_MDF, OFe_MDF, OH_PDF, FeH_PDF, OFe_PDF = ppc(p0_dict_list, par, bins, eri_ii_mdf)
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
            bins=bins,
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
np.savez('./SFE_SFH_eta_MDF.npz', **results)

if plot_pocomc:
    # Plot Run Diagnostics & Trace
    pc.plotting.run(results);
    pc.plotting.trace(
        results,
        labels=gal_par_names+[f'star_{i+1:02.0f}' for i in range(n_obj)] if use_obs_errs else gal_par_names,
    );
if plot_corner:
    # Plot Corner
    flat_samples = results['samples']
    fig = plt.figure(figsize=(20,20))
    fig = corner(
        flat_samples[:, :len(gal_par_names)],
        labels=gal_par_names,
        quantiles=(0.16, 0.5, 0.84),
        fig=fig,
        show_titles=True,
        title_kwargs=dict(fontsize=24),
        label_kwargs=dict(fontsize=24),
        hist_kwargs=dict(density=True)
    );
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=18)
    # Overplot Priors
    axes = np.array(fig.axes).reshape((len(gal_par_names), len(gal_par_names)))
    for i, label in enumerate(gal_par_names):
        ax = axes[i, i]
        x = np.linspace(0.75*flat_samples[:, i].min(), 1.25*flat_samples[:, i].max(), 10000)
        ax.plot(x, np.exp(priors[label](x)), color="g", label='prior' if i == 0 else '')
        if i == 0:
            ax.legend(fontsize=24)
    plt.show()
if plot_ppc:
    flat_samples = results['samples']
    ppc_samples_dict = [{par_name: flat_samples[i,j] for j, par_name in enumerate(gal_par_names)} for i in range(flat_samples.shape[0])]
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