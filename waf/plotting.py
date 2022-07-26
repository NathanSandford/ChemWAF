import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from corner import corner


def plot_corner(samples, par_names, priors=None):
    fig = plt.figure(figsize=(15,15))
    fig = corner(
        samples[:, :len(par_names)],
        labels=par_names,
        quantiles=(0.16, 0.5, 0.84),
        fig=fig,
        show_titles=True,
        title_kwargs=dict(fontsize=24),
        label_kwargs=dict(fontsize=24),
        hist_kwargs=dict(density=True)
    );
    if priors is not None:
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=18)
            axes = np.array(fig.axes).reshape((len(par_names), len(par_names)))
        for i, label in enumerate(par_names):
            ax = axes[i, i]
            x = np.linspace(0.75*samples[:, i].min(), 1.25*samples[:, i].max(), 10000)
            ax.plot(x, np.exp(priors[label](x)), color="g", label='prior' if i == 0 else '')
            if i == 0:
                ax.legend(fontsize=24)
    return fig


def plot_latentFeH(samples, latentFeH_priors, mod_bins, n_gal_par, n_obj):
    prior_vals = np.exp(latentFeH_priors(mod_bins[:, np.newaxis]))
    n_col = 3
    n_row = n_obj // n_col + (n_obj % n_col > 0)
    fig = plt.figure(figsize=(8*n_col,1*n_obj))
    gs = GridSpec(n_row, n_col)
    gs.update(hspace=0.0, wspace=0.0)
    for i in range(n_obj):
        ax = plt.subplot(gs[i // n_col, i % n_col])
        ax.hist(
            samples[:, n_gal_par+i], bins=mod_bins,
            histtype='step', density=True, color='k', label=f'Star {i+1:02.0f}'
        )
        ax.plot(mod_bins, prior_vals[:, i], color='g', label='CaHK [Fe/H]' if i % n_col == 0 else '')
        ax.set_xlim(-4 - 0.2, -1 + 0.2)
        if i % n_col == 0:
            ax.set_ylabel('Probability', fontsize=24)
            ax.tick_params('y', labelsize=18)
        else:
            ax.tick_params('y', labelsize=0)
        if i // n_col == n_row - 1:
            ax.set_xlabel('[Fe/H]', fontsize=24)
            ax.tick_params('x', labelsize=18)
        else:
            ax.tick_params('x', labelsize=0)
        ax.legend(fontsize=24, loc='upper right')
    plt.tight_layout()
    return fig


def plot_FeH_MDF(samples, par_names, obs, obs_mdf, mod_mdf, use_obs_errs=False):
    fig = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    ax.scatter(obs['FeH'], np.ones_like(obs['FeH']), marker='|', c='k', s=100)
    ax.errorbar(0, 1, xerr=np.median(obs['dFeH']), fmt='ok', capsize=10)
    ax.text(0, 1.75, 'Median Error', fontsize=24, ha='center')
    if use_obs_errs:
        ppc_FeH_MDF = np.zeros((samples.shape[0], obs_mdf['bins'].shape[0] - 1))
        for i in range(samples.shape[0]):
            ppc_FeH_MDF[i], _ = np.histogram(samples[i, len(par_names):], bins=obs_mdf['bins'], density=False)
        ax.stairs(obs_mdf['counts'], obs_mdf['bins'], color='k', lw=3, ls='--', label='Fu+ 2022')
        ax.stairs(
            np.percentile(ppc_FeH_MDF, 50, axis=0),
            obs_mdf['bins'],
            color='k', lw=3, label='Latent [Fe/H]',
        )
        ax.stairs(
            np.percentile(ppc_FeH_MDF, 97.5, axis=0),
            obs_mdf['bins'],
            baseline=np.percentile(ppc_FeH_MDF, 2.5, axis=0),
            fill=True, alpha=0.2, color='k', label='95% CI',
        )
    else:
        ax.stairs(obs_mdf['counts'], obs_mdf['bins'], color='k', lw=3, label='Fu+ 2022')
    ax.stairs(
        np.percentile(mod_mdf, 50, axis=0),
        obs_mdf['bins'],
        color='r', lw=3, label='Median Prediction',
    )
    ax.stairs(
        np.percentile(mod_mdf, 97.5, axis=0),
        obs_mdf['bins'],
        baseline=np.percentile(mod_mdf, 2.5, axis=0),
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
    return fig


def plot_OH_MDF(samples, par_names, obs, obs_mdf, mod_mdf, use_obs_errs=False):
    fig = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    try:
        ax.scatter(obs['OH'], np.ones_like(obs['OH']), marker='|', c='k', s=100)
        ax.errorbar(0, 1, xerr=np.median(obs['dOH']), fmt='ok', capsize=10)
        ax.text(0, 1.75, 'Median Error', fontsize=24, ha='center')
        obs_o_mdf = True
    except KeyError:
        obs_o_mdf = False
        pass
    if use_obs_errs:
        ppc_OH_MDF = np.zeros((samples.shape[0], obs_mdf['bins'].shape[0] - 1))
        for i in range(samples.shape[0]):
            ppc_OH_MDF[i], _ = np.histogram(samples[i, len(par_names):], bins=obs_mdf['bins'], density=False)
        if obs_o_mdf:
            ax.stairs(obs_mdf['counts'], obs_mdf['bins'], color='k', lw=3, ls='--', label='Fu+ 2022')
        ax.stairs(
            np.percentile(ppc_OH_MDF, 50, axis=0),
            obs_mdf['bins'],
            color='k', lw=3, label='Latent [Fe/H]',
        )
        ax.stairs(
            np.percentile(ppc_OH_MDF, 97.5, axis=0),
            obs_mdf['bins'],
            baseline=np.percentile(ppc_FeH_MDF, 2.5, axis=0),
            fill=True, alpha=0.2, color='k', label='95% CI',
        )
    else:
        if obs_o_mdf:
            ax.stairs(obs_mdf['counts'], obs_mdf['bins'], color='k', lw=3, label='Fu+ 2022')
    ax.stairs(
        np.percentile(mod_mdf, 50, axis=0),
        obs_mdf['bins'],
        color='r', lw=3, label='Median Prediction',
    )
    ax.stairs(
        np.percentile(mod_mdf, 97.5, axis=0),
        obs_mdf['bins'],
        baseline=np.percentile(mod_mdf, 2.5, axis=0),
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
    return fig


def plot_evolution(t, FeH, OH, OFe, SFR):
    fig = plt.figure(figsize=(16,8))
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(t,  np.percentile(FeH, 50, axis=0), color='r', lw=3, label='[Fe/H]')
    ax1.fill_between(t, np.percentile(FeH, 2.5, axis=0), np.percentile(FeH, 97.5, axis=0), alpha=0.2, color='r',)
    ax1.plot(t,  np.percentile(OH, 50, axis=0), color='b', lw=3, label='[O/H]')
    ax1.fill_between(t, np.percentile(OH, 2.5, axis=0), np.percentile(OH, 97.5, axis=0), alpha=0.2, color='b',)
    ax1.plot(t,  np.percentile(OFe, 50, axis=0), color='purple', lw=3, label='[O/Fe]')
    ax1.fill_between(t, np.percentile(OFe, 2.5, axis=0), np.percentile(OFe, 97.5, axis=0), alpha=0.2, color='purple',)
    ax2.plot(t,  np.percentile(SFR, 50, axis=0), color='grey', lw=3, label='Fractional SFR')
    ax2.fill_between(t, np.percentile(SFR, 2.5, axis=0), np.percentile(SFR, 97.5, axis=0), alpha=0.2, color='grey',)
    ax1.legend(fontsize=24, loc='upper left')
    ax2.legend(fontsize=24, loc='lower left')
    ax1.set_xlabel('[X/Y]', fontsize=36)
    ax2.set_ylabel(r'dSFR/dt', fontsize=36)
    ax1.tick_params('x', labelsize=24)
    ax1.tick_params('y', labelsize=24)
    plt.tight_layout()
    return fig


def plot_tinsley(FeH, OFe, mod_bins):
    n_samples = FeH.shape[0]
    fig = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    for i in range(n_samples):
        ax.plot(FeH[i, 10::], OFe[i, 10::], lw=0.5, alpha=0.05, c='r')
    ax.set_ylabel('[O/Fe]', fontsize=36)
    ax.set_xlabel('[Fe/H]', fontsize=36)
    ax.set_xlim(-4.0, -1.0)
    ax.set_ylim(0.05, 0.35)
    ax.tick_params('x', labelsize=24)
    ax.tick_params('y', labelsize=24)
    plt.tight_layout()
    return fig
