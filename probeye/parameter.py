class Parameters(dict):

    @property
    def n_prms(self):
        return len(self)

    @property
    def n_calibration_prms(self):
        return len([name for name, prm in self.items()
                    if prm.role == "calibration"])

    @property
    def n_constant_prms(self):
        return len([name for name, prm in self.items()
                    if prm.role == "const"])


class Parameter:
    def __init__(self, prm_dict):
        self.index = prm_dict['index']
        self.type = prm_dict['type']
        self.role = prm_dict['role']
        self.prior = prm_dict['prior']
        self.value = prm_dict['value']
        self.info = prm_dict['info']
        self.tex = prm_dict['tex']
