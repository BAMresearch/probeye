class Parameters(dict):
    """
    The main parameter 'library'. In this dictionary, all of the problem's
    parameters are stored. The parameter's names are the keys, and the
    associated values are ParameterProperties-objects, see below.
    """
    def __setitem__(self, key, value):
        """Performs a type-check before adding a parameter to the dictionary."""
        if type(key) != str:
            raise ValueError(
                f"The key must be a parameters name (string), but you provided "
                f"something of type '{type(key)}'.")
        if type(value) != ParameterProperties:
            raise ValueError(
                f"The properties of your parameter must be given in form of an "
                f"ParameterProperties-object. But you provided something of "
                f"type '{type(value)}'.")
        super().__setitem__(key, value)

    @property
    def n_prms(self):
        """Access the number of all parameters as an attribute."""
        return len(self)

    @property
    def n_calibration_prms(self):
        """Access the number of all 'calibration'-parameters as an attribute."""
        return len([name for name, prm in self.items()
                    if prm.role == "calibration"])

    @property
    def n_constant_prms(self):
        """Access the number of all 'const'-parameters as an attribute."""
        return len([name for name, prm in self.items()
                    if prm.role == "const"])


class ParameterProperties:
    """
    Describes relevant properties of a ('calibration' or 'const') parameter.
    Objects from this class are associated with the parameter's name in the
    dictionary class 'Parameters', see above. The use of this class as opposed
    to a standard dictionary allows convenient auto-completion while coding.
    """
    def __init__(self, prm_dict):
        """
        Parameters
        ----------
        prm_dict : dict
            The keys are 'index', 'type', 'role', 'prior', 'value', 'info' and
            'tex', while the values are the corresponding values of these
            properties. See also the explanations in InferenceProblem.__init__()
            for more detailed information.
        """
        self.index = prm_dict['index']
        self.type = prm_dict['type']
        self.role = prm_dict['role']
        self.prior = prm_dict['prior']
        self.value = prm_dict['value']
        self.info = prm_dict['info']
        self.tex = prm_dict['tex']
