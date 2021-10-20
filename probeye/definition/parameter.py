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

    def add_parameter(self, prm_name, prm_type, const=None, prior=None,
                      info="No explanation provided", tex=None):
        """
        Adds a parameter ('const' or 'latent') to the Parameters-object. The
        main functionality of this method is to distinguish between the two
        types ('const' and 'latent') and in creating the prior-object and
        adding the prior-parameters when adding a latent param. to the problem.

        Parameters
        ----------
        prm_name : str
            The name of the parameter which should be added to the problem.
        prm_type : str
            Either 'model' (for a model parameter), 'prior' (for a prior
            parameter) or 'noise' (for a noise parameter).
        const : float or None, optional
            If the added parameter is a 'const'-parameter, the corresponding
            value has to be specified by this argument.
        prior : tuple or list of two elements or None, optional
            If the added parameter is a 'latent'-parameter, this argument
            has to be given as a 2-tuple. The first element (a string) defines
            the prior-type (will be referenced in inference routines). The 2nd
            element must be a dictionary stating the prior's parameters as keys
            and their numeric values as values or the name of a pre-defined
            parameter within the problem scope. An example for a normal prior:
            ('normal', {'loc': 0.0, 'scale': 1.0}). In order to define the
            prior's parameters, check out the prior definitions in priors.py.
        info : str, optional
            Short explanation on the added parameter.
        tex : str or None, optional
            The TeX version of the parameter's name, for example r'$\beta$'
            for a parameter named 'beta'.
        """

        # make sure the parameters has not been defined yet
        self.confirm_that_parameter_does_not_exists(prm_name)

        # if neither const nor prior are given, the parameter is interpreted as
        # being defined as latent with an uninformative prior
        if const is None and prior is None:
            prior = ('uninformative', {})

        # add the parameter to the central parameter dictionary
        if const is None:  # i.e. adding 'latent'-parameter
            # first, define the index of this parameter in the numeric vector
            # theta, which is given to self.loglike and self.logprior
            prm_index = self.n_latent_prms
            # the prm_value is reserved for 'const'-parameter; hence, it is set
            # to None in this case, where we are adding a 'latent'-param.
            prm_value = None
            # the remaining code in this if-branch defines the prior that is
            # associated with this 'latent'-parameter; first, however, check
            # whether the given prior has a valid structure
            if type(prior) not in [list, tuple]:
                raise TypeError(
                    f"The given prior is of type {type(prior)} but must be "
                    f"either a list or a tuple!")
            if len(prior) != 2:
                raise RuntimeError(
                    f"The given prior must be a list/tuple with two elements. "
                    f"However, the given prior has {len(prior)} element(s).")
            if type(prior[0]) is not str:
                raise TypeError(
                    f"The first element of the prior must be of type string. "
                    f"However, the given first element is of type "
                    f"{type(prior[0])}.")
            if type(prior[1]) is not dict:
                raise TypeError(
                    f"The second element of the prior must be of type dict. "
                    f"However, the given second element is of type "
                    f"{type(prior[1])}.")
            # extract the prior's elements
            prior_type = prior[0]  # e.g. 'normal', 'lognormal', etc.
            prior_dict = prior[1]  # dictionary with parameter-value pairs
            prior_parameter_names = []
            for prior_parameter_name, value in prior_dict.items():
                # create unique name for this prior parameter
                new_name = f"{prior_parameter_name}_{prm_name}"
                prior_parameter_names.append(new_name)
                if type(value) in {float, int}:
                    # in this case, the prior-parameter is considered a 'const'-
                    # parameter and added to the problem accordingly here
                    default_info = f"{prior_type.capitalize()} "
                    default_info += f"prior's parameter "
                    default_info += f"for latent parameter '{prm_name}'"
                    # the following call is recursive, but only with a depth of
                    # one, since the added parameter is a constant here
                    self.add_parameter(new_name, 'prior', const=value,
                                       info=default_info)
                elif type(value) is str:
                    # in this case the prior-parameter is defined as an already
                    # defined parameter with the name stated in value
                    self.confirm_that_parameter_exists(value)
                else:
                    raise TypeError(
                        f"The prior-parameter {new_name} is not assigned a "
                        f"float, int or str, but something of type "
                        f"{type(value)}.")
            prior_name = f"{prm_name}_{prior_type}"  # unique name of this prior
            prm_prior = PriorBase(prm_name, prior_parameter_names,
                                  prior_name, prior_type)

        else:
            # in this case we are adding a 'const'-parameter, which means that
            # the prm_index and prm_prior values are not used here
            prm_index = None
            prm_prior = None
            prm_value = const

        # add the parameter to the central parameter dictionary
        self[prm_name] = ParameterProperties({'index': prm_index,
                                              'type': prm_type,
                                              'prior': prm_prior,
                                              'value': prm_value,
                                              'info': info,
                                              'tex': tex})

    def __setitem__(self, key, value):
        """Performs some checks before adding a parameter to the dictionary."""
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

    def __delitem__(self, key):
        """
        Deletes an item from itself while taking care of additional actions. For
        example removing prior-parameters when deleting a latent parameter or
        keeping the index-attributes of the latent parameters consistent.
        """

        # the given key is a parameter's name
        # (renaming for easier readability)
        prm_name = key

        # check if the given parameter exists
        self.confirm_that_parameter_exists(prm_name)

        # different steps must be taken depending on whether the parameter which
        # should be removed is a 'const'- or a 'latent'-parameter
        if self[prm_name].index is None:
            # in this case prm_name refers to a constant parameter; hence, we
            # can simply remove this parameter without having to take care of
            # other things as we will have to do for latent parameters
            dict.__delitem__(self, key)
        else:
            # in this case prm_name refers to a latent parameter; hence we need
            # to also remove the prior-parameters; also, we have to correct the
            # index values of the remaining latent parameters
            for prior_prm in self[prm_name].prior.hyperparameters.keys():
                self.__delitem__(prior_prm)  # recursive call
            dict.__delitem__(self, prm_name)
            # correct the indices of the remaining 'latent'-parameters; note
            # that the way how the correction is done is due to the fact that
            # the parameter.index attribute is protected, and cannot be changed
            # directly from outside
            idx_dict = {}
            idx = 0
            for prm_name, parameter in self.items():
                if parameter.index is not None:
                    idx_dict[prm_name] = idx
                    idx += 1
            for prm_name, idx in idx_dict.items():
                self[prm_name] = self[prm_name].changed_copy(index=idx)

    def confirm_that_parameter_exists(self, prm_name):
        """
        Checks if a parameter, given by its name, exists among the currently
        defined parameters. An error is raised when the given parameter does not
        exist yet.

        Parameters
        ----------
        prm_name : str
            A global parameter name.
        """
        if prm_name not in self:
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet.")

    def confirm_that_parameter_does_not_exists(self, prm_name):
        """
        Checks if a parameter, given by its name, exists among the currently
        defined parameters. An error is raised when the given parameter does
        already exist.

        Parameters
        ----------
        prm_name : str
            A global parameter name.
        """
        if prm_name in self:
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has already been defined.")

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
        return [name for name, prm in self.items() if prm.is_latent]

    @property
    def n_latent_prms(self):
        """Access the number of all 'latent'-parameters as an attribute."""
        return len(self.latent_prms)

    @property
    def constant_prms(self):
        """Access the names of all 'const'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.is_const]

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

        # whitespace in the tex strings is a problem for some plotting routines,
        # so they are replaced here by a math-command for whitespace that does
        # not contain actual whitespace
        if self.tex:
            self.tex = self.tex.replace(" ", r'$\enspace$')

        # check the given values
        self.check_consistency()

    # noinspection PyShadowingBuiltins
    def changed_copy(self, index=None, type=None, prior=None, value=None,
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
    def is_latent(self):
        """Adds a pseudo-attribute self.is_latent, which allows a convenient
           check on whether a parameter is latent or not."""
        return self._index is not None

    @property
    def is_const(self):
        """Adds a pseudo-attribute self.is_const, which allows a convenient
           check on whether a parameter is constant or not."""
        return not self.is_latent

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
