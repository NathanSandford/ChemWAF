import numpy as np


def waf2017(
    t: np.ndarray,
    tauSFE: float = 2.0,
    tauSFH: float = 8.0,
    yAlphaCC: float = 0.015,
    yFeCC: float = 0.00015,
    yFeIa: float = 0.0013,
    fRetCC: float = 1.,
    fRetIa: float = 1.,
    r: float = 0.4,
    eta: float = 2.0,
    tauIa: float = 1.5,
    tDminIa: float = 0.05,
    SolarAlpha: float = 0.0056,
    SolarFe: float = 0.0012,
    SFH_fn: str = 'exponential',
    IaDTD_fn: str = 'exponential',
):
    '''
    Compute [Alpha/H], [Fe/H], [Alpha/Fe] tracks using WAF analytic model for constant SFR or exponentially declining SFR.
    By default, the model assumes alpha == oxygen.
    Reference: Weinberg, Andrews, & Freudenburg 2017, ApJ 837, 183 (Particularly Appendix C)

    :param np.ndarray t: time array (in Gyr) for desired outputs
    :param float tauSFE: SFE efficiency timescale [2.0]
    :param float tauSFH: e-folding timescale of SFH [8.0]
        (Ignored if SFH_fn == 'constant')
    :param float yAlphaCC: IMF-averaged CCSN alpha yield [0.015]
    :param float yFeCC: IMF-averaged CCSN iron yield [0.0015]
    :param float yFeIa: IMF-averaged SNIa iron yield over 10 Gyr [0.0013]
    :param float fRetCC:
    :param float fRetIa:
    :param float r: recycling fraction [0.4, based on Kroupa IMF]
    :param float eta: outflow mass loading factor [2.0]
    :param float tauIa: = e-folding time for SNIa DTD [1.5]
        (Ignored if IaDTD_fn == 'powerlaw')
    :param float tDminIa: minimum time delay for SNIa [0.05]
        (If IaDTD_fn == 'powerlaw', then tDminIa must be either 0.05 or 0.15)
    :param float SolarAlpha: solar alpha mass fraction [0.0056]
    :param float SolarFe: solar iron mass fraction [0.0012]
    :param str SFH_fn: functional form of the SFH. ['exponential']
        Must be one of 'constant', 'exponential', or 'linexp'
    :param IaDTD_fn: functional form of the SN Ia DTD ['exponential]
        Must be either 'exponential' or 'powerlaw'
    :return Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        SFR, AlphaH, FeH, AlphaFe --- each a 1-d array corresponding to the time array, t
        (SFR is normalized such that for evenly spaced t, it is the fraction of stars formed in each time step.)
    '''
    # Prophylactically preventing divide-by-zero errors
    xfac = 1.0013478
    yfac = 1.0016523
    # Modulate Yields by Return Fraction
    yAlphaCC *= fRetCC
    yFeCC *= fRetCC
    yFeIa *= fRetIa
    # Parse Ia DTD
    if IaDTD_fn == 'exponential':
        tauIa1 = tauIa
        tauIa2 = tauIa
        Ia_norm1 = 1.0
        Ia_norm2 = 0.0
    elif IaDTD_fn == 'powerlaw':
        if tDminIa == 0.05:
            tauIa1 = 0.25
            tauIa2 = 3.5
            Ia_norm1 = 0.493
            Ia_norm2 = 0.507
        elif tDminIa == 0.15:
            tauIa1 = 0.5
            tauIa2 = 5.0
            Ia_norm1 = 0.478
            Ia_norm2 = 0.522
        else:
            RuntimeError("tDminIa must be either 0.05 or 0.15 if IaDTD_fn == 'powerlaw' ")
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
    ZAlphaEq = yAlphaCC * tauDepSFH / tauSFE
    ZFeCCEq = yFeCC * tauDepSFH / tauSFE
    ZFeIaEq1 = Ia_norm1 * yFeIa * ((tauDepSFH / tauSFE) * (tauIaSFH1 / tauIa1) * np.exp(tDminIa / tauSFH))
    ZFeIaEq2 = Ia_norm2 * yFeIa * ((tauDepSFH / tauSFE) * (tauIaSFH2 / tauIa2) * np.exp(tDminIa / tauSFH))
    # Compute non-equilibrium abundances
    if SFH_fn in ['constant', 'exponential']:  # WAF equations 50, 52, and 53
        delta_t = t - tDminIa
        ZAlpha = ZAlphaEq * (1. - np.exp(-t / tauDepSFH))
        ZFeCC = ZAlpha * ZFeCCEq / ZAlphaEq
        ZFeIa1 = ZFeIaEq1 * (1. - np.exp(-delta_t / tauDepSFH) - (tauDepIa1 / tauDepSFH) * (
                    np.exp(-delta_t / tauIaSFH1) - np.exp(-delta_t / tauDepSFH)))
        ZFeIa2 = ZFeIaEq2 * (1. - np.exp(-delta_t / tauDepSFH) - (tauDepIa2 / tauDepSFH) * (
                    np.exp(-delta_t / tauIaSFH2) - np.exp(-delta_t / tauDepSFH)))
    elif SFH_fn == 'linexp':  # WAF equations 56-58
        delta_t = t - tDminIa
        ZAlpha = ZAlphaEq * (1. - (tauDepSFH / t) * (1. - np.exp(-t / tauDepSFH)))
        ZFeCC = ZAlpha * ZFeCCEq / ZAlphaEq
        ZFeIa1 = ZFeIaEq1 * (tauIaSFH1 / t) * (
                delta_t / tauIaSFH1
                + (tauDepIa1 / tauDepSFH) * np.exp(-delta_t / tauIaSFH1)
                + (1. + (tauDepSFH / tauIaSFH1) - (tauDepIa1 / tauDepSFH)) * np.exp(-delta_t / tauDepSFH)
                - (1. + (tauDepSFH / tauIaSFH1))
        )
        ZFeIa2 = ZFeIaEq2 * (tauIaSFH2 / t) * (
                delta_t / tauIaSFH2
                + (tauDepIa2 / tauDepSFH) * np.exp(-delta_t / tauIaSFH2)
                + (1. + (tauDepSFH / tauIaSFH2) - (tauDepIa2 / tauDepSFH)) * np.exp(-delta_t / tauDepSFH)
                - (1. + (tauDepSFH / tauIaSFH2))
        )
        ZFeIa1[t < tDminIa] = 0
        ZFeIa2[t < tDminIa] = 0
    else:
        RuntimeError("SFH_fn must be one of 'constant', 'exponential', or 'linexp' ")
    Zfe = ZFeCC + ZFeIa1 + ZFeIa2
    # Compute SFR
    if SFH_fn == 'constant':
        SFR = np.ones_like(t)
    elif SFH_fn == 'exponential':
        SFR = np.exp(-t / tauSFH)
    elif SFH_fn == 'linexp':
        SFR = t * np.exp(-t / tauSFH)
    else:
        RuntimeError("SFH_fn must be one of 'constant', 'exponential', or 'linexp' ")
    SFR /= np.sum(SFR)
    # Convert to [Alpha/H], [Fe/H]
    AlphaH = np.log10(ZAlpha / SolarAlpha)
    FeH = np.log10(Zfe / SolarFe)
    AlphaFe = AlphaH - FeH
    return SFR, AlphaH, FeH, AlphaFe
