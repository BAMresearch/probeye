# third party imports
from tabulate import tabulate

# local imports
from probeye.subroutines import titled_table, simplified_list_string

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
    def prms(self):
        """Access the names of all parameters as an attribute."""
        return [*self.keys()]

    @property
    def n_prms(self):
        """Access the number of all parameters as an attribute."""
        return len(self)

    @property
    def latent_prms(self):
        """Access the names of all 'latent'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.role == "latent"]

    @property
    def n_latent_prms(self):
        """Access the number of all 'latent'-parameters as an attribute."""
        return len(self.latent_prms)

    @property
    def constant_prms(self):
        """Access the names of all 'const'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.role == "const"]

    @property
    def n_constant_prms(self):
        """Access the number of all 'const'-parameters as an attribute."""
        return len(self.constant_prms)

    @property
    def model_prms(self):
        """Access the names of all 'model'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.type == "model"]

    @property
    def n_model_prms(self):
        """Access the number of all 'model'-parameters as an attribute."""
        return len(self.model_prms)

    @property
    def prior_prms(self):
        """Access the names of all 'prior'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.type == "prior"]

    @property
    def n_prior_prms(self):
        """Access the number of all 'prior'-parameters as an attribute."""
        return len(self.prior_prms)

    @property
    def noise_prms(self):
        """Access the names of all 'noise'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.type == "noise"]

    @property
    def n_noise_prms(self):
        """Access the number of all 'prior'-parameters as an attribute."""
        return len(self.noise_prms)

    def parameter_overview(self, tablefmt="presto"):
        """
        Returns a string providing an overview of the defined parameters.

        Parameters
        ----------
        tablefmt : string, optional
            An argument for the tabulate function defining the style of the
            generated table. Check out tabulate's documentation for more info.

        Returns
        -------
        prm_string : string
            This string describes a nice table with some essential information
            on the parameters of the problem.
        """
        # each element describes one row in the table to be generated
        rows = [('Model parameters',
                simplified_list_string(self.model_prms),
                self.n_model_prms),
                ('Prior parameters',
                simplified_list_string(self.prior_prms),
                self.n_prior_prms),
                ('Noise parameters',
                simplified_list_string(self.noise_prms),
                self.n_noise_prms),
                ('Const parameters',
                simplified_list_string(self.constant_prms),
                self.n_constant_prms),
                ('Latent parameters',
                simplified_list_string(self.latent_prms),
                self.n_latent_prms)]
        # these are the strings appearing in the column headers
        headers = ["Parameter type/role", "Parameter names", "Count"]
        prm_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prm_string = titled_table('Parameter overview', prm_table)
        return prm_string

    def parameter_explanations(self, tablefmt="presto"):
        """
        Returns a string providing short explanations on the defined parameters.

        Parameters
        ----------
        tablefmt : string
            An argument for the tabulate function defining the style of the
            generated table. Check out tabulate's documentation for more info.

        Returns
        -------
        prm_string : string, optional
            This string describes a nice table with short explanations on the
            parameters of the problem.
        """
        rows = [(name, prm.info) for name, prm in self.items()]
        headers = ["Name", "Short explanation"]
        prm_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prm_string = titled_table('Parameter explanations', prm_table)
        return prm_string

    def const_parameter_values(self, tablefmt="presto"):
        """
        Returns a string providing the values of the defined 'const'-parameters.

        Parameters
        ----------
        tablefmt : string
            An argument for the tabulate function defining the style of the
            generated table. Check out tabulate's documentation for more info.

        Returns
        -------
        prm_string : string, optional
            This string describes a nice table with the names and values of the
            constant parameters of the problem.
        """
        rows = [(name, prm.value)
                for name, prm in self.items() if prm.value is not None]
        headers = ["Name", "Value"]
        prm_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prm_string = titled_table('Constant parameters', prm_table)
        return prm_string

class ParameterProperties:
    """
    Describes relevant properties of a ('latent' or 'const') parameter.
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
        self._index = prm_dict['index']
        self._type = prm_dict['type']
        self._role = prm_dict['role']
        self._prior = prm_dict['prior']
        self._value = prm_dict['value']
        self._info = prm_dict['info']
        self._tex = prm_dict['tex']

    @property
    def index(self):
        """Access self._index from outside via self.index."""
        return self._index

    @index.setter
    def index(self, value):
        """Raise a specific error when trying to directly set self.index."""
        raise AttributeError(
            "Changing a parameter's index directly is prohibited!")

    @property
    def type(self):
        """Access self._type from outside via self.type."""
        return self._type

    @type.setter
    def type(self, value):
        """Raise a specific error when trying to directly set self.type."""
        raise AttributeError(
            "Changing a parameter's type directly is prohibited!")

    @property
    def role(self):
        """Access self._role from outside via self.role."""
        return self._role

    @role.setter
    def role(self, value):
        """Raise a specific error when trying to directly set self.role."""
        raise AttributeError(
            "You cannot change a parameter's role directly! Use "
            "InferenceProblem.change_parameter_role instead.")

    @property
    def prior(self):
        """Access self._prior from outside via self.prior."""
        return self._prior

    @prior.setter
    def prior(self, value):
        """Raise a specific error when trying to directly set self.prior."""
        raise AttributeError(
            "Changing a parameter's prior directly is prohibited!")

    @property
    def value(self):
        """Access self._value from outside via self.value."""
        return self._value

    @value.setter
    def value(self, value):
        """Raise a specific error when trying to directly set self.value."""
        raise AttributeError(
            "Changing a parameter's value directly is prohibited!")

    @property
    def info(self):
        """Access self._info from outside via self.info."""
        return self._info

    @info.setter
    def info(self, value):
        """Raise a specific error when trying to directly set self.info."""
        raise AttributeError(
            "You cannot change a parameter's info-property directly! Use "
            "InferenceProblem.change_parameter_info instead.")

    @property
    def tex(self):
        """Access self._tex from outside via self.tex."""
        return self._tex

    @tex.setter
    def tex(self, value):
        """Raise a specific error when trying to directly set self.tex."""
        raise AttributeError(
            "You cannot change a parameter's tex-property directly! Use "
            "InferenceProblem.change_parameter_info instead.")
