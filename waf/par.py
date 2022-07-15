class DefaultParSet:
    def __init__(
        self,
    ):
        self.tauSFE = 2.0
        self.tauSFH = 8.0
        self.mocc = 0.015
        self.mfecc = 0.0015
        self.mfeIa = 0.0013
        self.fRetCC = 1.
        self.fRetIa = 1.
        self.r = 0.4
        self.eta = 2.0
        self.tauIa = 1.5
        self.tdminIa = 0.05
        self.SolarO = 0.0056
        self.SolarFe = 0.0012
        self.SFH_fn = 'exponential'
        self.IaDTD_fn = 'exponential'

    def update(self, p):
        for p_name, p_val in p.items():
            if p_name == 'fRet':
                setattr(self, 'fRetCC', p_val)
                setattr(self, 'fRetIa', p_val)
            elif p_name[:3] == 'log':
                setattr(self, p_name[3:], 10 ** p_val)
            else:
                setattr(self, p_name, p_val)


class EriIIParSet(DefaultParSet):
    def __init__(
        self
    ):
        DefaultParSet.__init__(self)
