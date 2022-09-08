import numpy as np

class DefaultParSet:
    def __init__(
        self,
    ):
        self.tauSFE = 2.0
        self.tauSFH = 8.0
        self.yAlphaCC = 0.015
        self.yFeCC = 0.0015
        self.yFeIa = 0.0013
        self.fRetCC = 1.
        self.fRetIa = 1.
        self.r = 0.4
        self.eta = 2.0
        self.tauIa = 1.5
        self.tDminIa = 0.05
        self.SolarAlpha = 0.0056
        self.SolarFe = 0.0012
        self.SFH_fn = 'exponential'
        self.IaDTD_fn = 'exponential'
        self.dt = 0.001
        self.t_trunc = 12.0
        self.t = np.arange(self.dt, self.t_trunc + self.dt, self.dt)
        self.model_kwargs = dict(
            tauSFE=self.tauSFE,
            tauSFH=self.tauSFH,
            yAlphaCC=self.yAlphaCC,
            yFeCC=self.yFeCC,
            yFeIa=self.yFeIa,
            fRetCC=self.fRetCC,
            fRetIa=self.fRetIa,
            r=self.r,
            eta=self.eta,
            tauIa=self.tauIa,
            tDminIa=self.tDminIa,
            SolarAlpha=self.SolarAlpha,
            SolarFe=self.SolarFe,
            SFH_fn=self.SFH_fn,
            IaDTD_fn=self.IaDTD_fn,
            t=self.t,
        )

    def update(self, p):
        for p_name, p_val in p.items():
            if p_name == 'fRet':
                setattr(self, 'fRetCC', p_val)
                setattr(self, 'fRetIa', p_val)
            elif p_name[:3] == 'log':
                setattr(self, p_name[3:], 10 ** p_val)
            else:
                setattr(self, p_name, p_val)
        self.t = np.arange(self.dt, self.t_trunc + self.dt, self.dt)
        self.model_kwargs = dict(
            tauSFE=self.tauSFE,
            tauSFH=self.tauSFH,
            yAlphaCC=self.yAlphaCC,
            yFeCC=self.yFeCC,
            yFeIa=self.yFeIa,
            fRetCC=self.fRetCC,
            fRetIa=self.fRetIa,
            r=self.r,
            eta=self.eta,
            tauIa=self.tauIa,
            tDminIa=self.tDminIa,
            SolarAlpha=self.SolarAlpha,
            SolarFe=self.SolarFe,
            SFH_fn=self.SFH_fn,
            IaDTD_fn=self.IaDTD_fn,
            t=self.t,
        )
