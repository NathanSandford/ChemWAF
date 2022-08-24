import numpy as np


def waf2017(
    t: np.ndarray,
    tauSFE: float = 2.0,
    tauSFH: float = 8.0,
    mocc: float = 0.015,
    mfecc: float = 0.00015,
    mfeIa: float = 0.0013,
    fRetCC: float = 1.,
    fRetIa: float = 1.,
    r: float = 0.4,
    eta: float = 2.0,
    tauIa: float = 1.5,
    tdminIa: float = 0.05,
    SolarO: float = 0.0056,
    SolarFe: float = 0.0012,
    SFH_fn: str = 'exponential',
    IaDTD_fn: str = 'exponential',
):
    '''
    Compute [O/H], [Fe/H], [O/Fe] tracks using WAF analytic model for constant SFR or exponentially declining SFR
    Reference: Weinberg, Andrews, & Freudenburg 2017, ApJ 837, 183 (Particularly Appendix C)

    :param np.ndarray t: time array (in Gyr) for desired outputs
    :param float tauSFE: SFE efficiency timescale [2.0]
    :param float tauSFH: e-folding timescale of SFH [8.0]
        (Ignored if SFH_fn == 'constant')
    :param float mocc: IMF-averaged CCSN oxygen yield [0.015]
    :param float mfecc: IMF-averaged CCSN iron yield [0.0015]
    :param float mfeIa: IMF-averaged SNIa iron yield over 10 Gyr [0.0013]
    :param float fRetCC:
    :param float fRetIa:
    :param float r: recycling fraction [0.4, based on Kroupa IMF]
    :param float eta: outflow mass loading factor [2.0]
    :param float tauIa: = e-folding time for SNIa DTD [1.5]
        (Ignored if IaDTD_fn == 'powerlaw')
    :param float tdmin: minimum time delay for SNIa [0.05]
        (If IaDTD_fn == 'powerlaw', then tdmin must be either 0.05 or 0.15)
    :param float SolarO: solar oxygen mass fraction [0.0056]
    :param float SolarFe: solar iron mass fraction [0.0012]
    :return Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        SFR, OH, FeH, OFe --- each a 1-d array corresponding to the time array, t
        (SFR is normalized such that for evenly spaced t, it is the fraction of stars formed in each time step.)
    '''
    # Prophylactically preventing divide-by-zero errors
    xfac = 1.0013478
    yfac = 1.0016523
    # Modulate Yields by Return Fraction
    mocc *= fRetCC
    mfecc *= fRetCC
    mfeIa *= fRetIa
    # Parse Ia DTD
    if IaDTD_fn == 'exponential':
        tauIa1 = tauIa
        tauIa2 = tauIa
        Ia_norm1 = 1.0
        Ia_norm2 = 0.0
    elif IaDTD_fn == 'powerlaw':
        if tdminIa == 0.05:
            tauIa1 = 0.25
            tauIa2 = 3.5
            Ia_norm1 = 0.493
            Ia_norm2 = 0.507
        elif tdminIa == 0.15:
            tauIa1 = 0.5
            tauIa2 = 5.0
            Ia_norm1 = 0.478
            Ia_norm2 = 0.522
        else:
            RuntimeError("tdminIa must be either 0.05 or 0.15 if IaDTD_fn == 'powerlaw' ")
    else:
        raise RuntimeError("IaDTD_fn must be either 'exponential' or 'powerlaw' ")
    # Compute "Harmonic Difference" Timescales
    tauSFE *= xfac
    tauSFH *= yfac
    tauDep = tauSFE / (1 + eta - r)
    tauDepIa1 = 1. / (1. / tauDep - 1. / tauIa1)
    tauDepIa2 = 1. / (1. / tauDep - 1. / tauIa2)
    if SFH_fn == 'constant':
        tauDepSFH = tauDep
        tauIaSFH1 = tauIa1
        tauIaSFH2 = tauIa2
        tauSFH = 1e8
    elif SFH_fn in ['exponential', 'linexp']:
        tauDepSFH = 1. / (1. / tauDep - 1. / tauSFH)
        tauIaSFH1 = 1. / (1. / tauIa1 - 1. / tauSFH)
        tauIaSFH2 = 1. / (1. / tauIa2 - 1. / tauSFH)
    else:
        RuntimeError("SFH_fn must be one of 'constant', 'exponential', or 'linexp' ")
    # Compute equilibrium abundances, WAF equations 28-30
    ZoEq = mocc * tauDepSFH / tauSFE
    ZfeccEq = mfecc * tauDepSFH / tauSFE
    ZfeIaEq1 = Ia_norm1 * mfeIa * ((tauDepSFH / tauSFE) * (tauIaSFH1 / tauIa1) * np.exp(tdminIa / tauSFH))
    ZfeIaEq2 = Ia_norm2 * mfeIa * ((tauDepSFH / tauSFE) * (tauIaSFH2 / tauIa2) * np.exp(tdminIa / tauSFH))
    # Compute nonequilibrium abundances
    if SFH_fn in ['constant', 'exponential']:  # WAF equations 50, 52, and 53
        deltat = t - tdminIa
        Zo = ZoEq * (1. - np.exp(-t / tauDepSFH))
        Zfecc = Zo * ZfeccEq / ZoEq
        ZfeIa1 = ZfeIaEq1 * (1. - np.exp(-deltat / tauDepSFH) - (tauDepIa1 / tauDepSFH) * (
                    np.exp(-deltat / tauIaSFH1) - np.exp(-deltat / tauDepSFH)))
        ZfeIa2 = ZfeIaEq2 * (1. - np.exp(-deltat / tauDepSFH) - (tauDepIa2 / tauDepSFH) * (
                    np.exp(-deltat / tauIaSFH2) - np.exp(-deltat / tauDepSFH)))
    elif SFH_fn == 'linexp':  # WAF equations 56-58
        deltat = t - tdminIa
        Zo = ZoEq * (1. - (tauDepSFH / t) * (1. - np.exp(-t / tauDepSFH)))
        Zfecc = Zo * ZfeccEq / ZoEq
        ZfeIa1 = ZfeIaEq1 * (tauIaSFH1 / t) * (
                deltat / tauIaSFH1 \
                + (tauDepIa1 / tauDepSFH) * np.exp(-deltat / tauIaSFH1) \
                + (1. + (tauDepSFH / tauIaSFH1) - (tauDepIa1 / tauDepSFH)) * np.exp(-deltat / tauDepSFH) \
                - (1. + (tauDepSFH / tauIaSFH1))
        )
        ZfeIa2 = ZfeIaEq2 * (tauIaSFH2 / t) * (
                deltat / tauIaSFH2
                + (tauDepIa2 / tauDepSFH) * np.exp(-deltat / tauIaSFH2) \
                + (1. + (tauDepSFH / tauIaSFH2) - (tauDepIa2 / tauDepSFH)) * np.exp(-deltat / tauDepSFH) \
                - (1. + (tauDepSFH / tauIaSFH2))
        )
        ZfeIa1[t < tdminIa] = 0
        ZfeIa2[t < tdminIa] = 0
    else:
        RuntimeError("SFH_fn must be one of 'constant', 'exponential', or 'linexp' ")
    Zfe = Zfecc + ZfeIa1 + ZfeIa2
    # Compute SFR
    if SFH_fn == 'constant':
        SFR = np.ones_like(t)
    elif SFH_fn == 'exponential':
        SFR = np.exp(-t / tauSFH)
    elif SFH_fn == 'linexp':
        SFR = t * np.exp(-t / tauSFH)
    SFR /= np.sum(SFR)
    # Convert to [O/H], [Fe/H]
    OH = np.log10(Zo / SolarO)
    FeH = np.log10(Zfe / SolarFe)
    OFe = OH - FeH
    return SFR, OH, FeH, OFe
