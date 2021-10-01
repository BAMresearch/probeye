# third party imports
from tabulate import tabulate

# local imports
from probeye.subroutines import titled_table, simplified_list_string
from probeye.definition.prior import PriorBase

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
        # write attributes
        self._index = prm_dict['index']
        self._type = prm_dict['type']
        self._prior = prm_dict['prior']
        self._value = prm_dict['value']
        self.info = prm_dict['info']
        self.tex = prm_dict['tex']

        # check the given values
        self.check_consistency()

    # noinspection PyShadowingBuiltins
    def change(self, index=None, type=None, prior=None, value=None,
                  info=None, tex=None):
        """
        Convenience method that simplifies changing the attributes of a
        ParameterProperties object based on creating a new instance. The reason
        for this approach is that some of the attributes are private, and cannot
        (or at least should not) be changed directly from outside.

        See the explanations in InferenceProblem.__init__() for more detailed 
        information on the arguments.
        """
        return ParameterProperties({
                    "index": index if index is not None else self._index,
                    "type": type or self._type,
                    "prior": prior or self._prior,
                    "value": value or self._value,
                    "info": info or self.info,
                    "tex": tex or self.tex})

    def check_consistency(self):
        """
        Checks the defined attributes in both isolated checks (each attribute
        is checked without considering others) and cross-checks, where the
        combination of attributes is checked on consistency.
        """

        # ------------------------------- #
        #         Isolated checks         #
        # ------------------------------- #

        if not (type(self._index) == int or self._index is None):
            raise TypeError(
                f"Found invalid ParameterProperties._index attribute! It must "
                f"be of type int or None, but found {type(self._index)}.")

        if (self._index is not None) and (self._index < 0):
            raise RuntimeError(
                f"Found negative value for ParameterProperties._index! This "
                f"attribute must be a non-negative integer, but found a value "
                f"of {self._index}.")

        if type(self._type) != str:
            raise TypeError(
                f"Found invalid ParameterProperties._type attribute! Its type "
                f"must be str, but found {type(self._type)}.")

        if self._type not in ['model', 'prior', 'noise']:
            raise RuntimeError(
                f"Found invalid ParameterProperties._type attribute! It can "
                f"only assume the three values 'model', 'prior' or 'noise' but "
                f"found '{self._type}'.")

        if not (type(self._prior) == PriorBase or self._prior is None):
            raise TypeError(
                f"Found invalid ParameterProperties._prior attribute! It must "
                f"be of type PriorBase or None, but found {type(self._prior)}.")

        if not (type(self._value) in [float, int] or self._value is None):
            raise TypeError(
                f"Found invalid ParameterProperties._value attribute! It must "
                f"be of type float/int or None, but found {type(self._value)}.")

        if type(self.info) != str:
            raise TypeError(
                f"Found invalid ParameterProperties.info attribute! Its type "
                f"must be str, but found {type(self.info)}.")

        if not (type(self.tex) == str or self.tex is None):
            raise TypeError(
                f"Found invalid ParameterProperties.tex attribute! It must be "
                f"of type str or None, but found {type(self.tex)}.")

        # -------------------------------- #
        #           Cross checks           #
        # -------------------------------- #

        if self._index is not None:
            # in this case, we have a latent parameter
            if self._value is not None:
                raise RuntimeError(
                    f"ParameterProperties._index and ParameterProperties._value"
                    f" are both given (_index={self._index} and _value="
                    f"{self._value}), but one of them must be None!")
            if self._prior is None:
                raise RuntimeError(
                    f"ParameterProperties._index and ParameterProperties._prior"
                    f" are both given (_index={self._index} and prior="
                    f"{self._prior}), but one of them must be None!")

        else:
            # in this case, we have a constant parameter
            if self._value is None:
                raise RuntimeError(
                    f"ParameterProperties._index and ParameterProperties._value"
                    f" are both None, but one of them must be not None!")
            if self._prior is not None:
                raise RuntimeError(
                    f"ParameterProperties._index is None while Parameter"
                    f"Properties._prior is given ({self._prior}). This "
                    f"combination is not valid. Either the index must also be "
                    f"given, or the prior must also be None.")

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
        """Adds a pseudo-attribute self.role, which allows a convenient check
           on whether a parameter is latent or not."""
        return 'latent' if self._index is not None else 'const'

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
