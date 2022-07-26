from pathlib import Path
import numpy as np
import pandas as pd
import multiprocess as mp
import pocomc as pc
from waf.par import DefaultParSet
from waf.fitting.priors import UniformLogPrior, GaussianLogPrior
from waf.fitting.pocomc_unbinned import log_prior
from waf.fitting.pocomc_unbinned import log_likelihood
from waf.fitting.pocomc_unbinned import log_probability
from waf.fitting.pocomc_unbinned import ppc
import matplotlib as mpl
import matplotlib.pyplot as plt
from corner import corner
np.seterr(all="ignore");

nwalkers = 5000
data_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/data/EriII_MDF.dat')
output_file = Path('/global/scratch/users/nathan_sandford/ChemEv/EriII/samples/EriII_Unbinned.npz')

# Load Observed Data
obs = pd.read_csv(data_file, index_col=0)
obs_bins = np.linspace(-10, 2.0, 49)
counts, obs_bins = np.histogram(obs['FeH'], bins=obs_bins, density=False)
obs_mdf = {
    'counts': counts,
    'bins': obs_bins,
}
n_star = obs.shape[0]
# Load Default Parameters
par = DefaultParSet()
par.t = np.arange(0.0001, 1.0001, 0.0001)
# Define Priors
priors = dict(
    logtauSFE=UniformLogPrior('logtauSFE', 0, 4, -np.inf),
    tauSFH=GaussianLogPrior('tauSFH', 0.7, 0.2, 0, 1e2),
    eta=UniformLogPrior('eta', 0, 1e3, -np.inf),
    fRetCC=UniformLogPrior('fRetCC', 0, 1, -np.inf),
    fRetIa=UniformLogPrior('fRetIa', 0, 1, -np.inf),
)
gal_par_names = list(gal_priors.keys())
bounds = np.array([
    [1e-2, 4],  # logtauSFE
    [1e-2, 1e2],  # tauSFH
    [0, 1e3],  # eta
    [0, 1],  # fRetCC
    [0, 1],  # fRetIa
])
# Initialize Walkers
p0_list = []
while len(p0_list) < nwalkers:
    p = np.array(
    [
        2.0 + 1e-1 * np.random.randn(),  # logtauSFE
        0.70 + 1e-1 * np.random.randn(),  # tauSFH
        100 + 1e+1 * np.random.randn(),  # eta
        0.5 + 2e-1 * np.random.randn(),  # fRetCC
        0.5 + 2e-1 * np.random.randn(),  # fRetIa
    ])
    if fit_latent_FeH:
        p = np.concatenate([p, [obs['FeH'][i] + obs['dFeH'][i] * np.random.randn() for i in range(n_star)]])
    if np.isfinite(
        log_probability(
            p,
            default_par=par,
            priors=priors,
            gal_par_names=gal_par_names,
            obs=obs,
        )
    ):
        p0_list.append(p)
p0 = np.vstack(p0_list)
nwalkers, ndim = p0.shape
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
            obs=obs,
        ),
        log_prior=log_prior,
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
