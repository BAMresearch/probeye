# standard library
from typing import Union, Optional, List
import re

# third party imports
from tabulate import tabulate
from loguru import logger
import numpy as np

# local imports
from probeye.subroutines import titled_table
from probeye.subroutines import simplified_list_string
from probeye.subroutines import len_or_one
from probeye.subroutines import translate_number_string
from probeye.subroutines import count_intervals
from probeye.subroutines import convert_to_tuple
from probeye.definition.prior import PriorBase
from probeye.definition.distribution import ProbabilityDistribution, Uninformative


class Parameters(dict):
    """
    The main parameter 'library'. In this dictionary, all of the problem's parameters
    are stored. The parameter's names are the keys, and the associated values are
    ParameterProperties-objects, see below.
    """

    def add_parameter(
        self,
        prm_name: str,
        prm_type: str = "not defined",
        dim: Optional[int] = 1,
        domain: str = "(-oo, +oo)",
        const: Union[int, float, tuple, np.ndarray, None] = None,
        prior: Optional[ProbabilityDistribution] = None,
        info: str = "No explanation provided",
        tex: Optional[str] = None,
    ):
        """
        Adds a parameter ('const' or 'latent') to the Parameters-object. The main
        functionality of this method is to distinguish between the two types ('const'
        and 'latent') and in creating the prior-object and adding the prior-parameters
        when adding a latent param. to the problem.

        Parameters
        ----------
        prm_name
            The name of the parameter which should be added to the problem.
        prm_type
            Either 'model' (for a model parameter), 'prior' (for a prior parameter) or
            'likelihood' (for a likelihood parameter).
        dim
            The parameter's dimension.
        domain
            The parameter's domain (i.e., values it may assume). Note that this argument
            is only considered for latent parameter, but not for a constant.
        const
            If the added parameter is a 'const'-parameter, the corresponding value has
            to be specified by this argument.
        prior
            If the added parameter is a 'latent'-parameter, this argument has to be
            given as a 2-tuple. The first element (a string) defines the prior-type
            (will be referenced in inference routines). The 2nd element must be a
            dictionary stating the prior's parameters as keys and their numeric values
            as values or the name of a pre-defined parameter within the problem scope.
            An example for a normal prior: ('normal', {'loc': 0.0, 'scale': 1.0}). In
            order to define the prior's parameters, check out the prior definitions in
            priors.py.
        info
            Short explanation on the added parameter.
        tex
            The TeX version of the parameter's name, for example r'$\beta$' for a
            parameter named 'beta'.
        """

        # make sure the parameters has not been defined yet
        self.confirm_that_parameter_does_not_exists(prm_name)

        # if neither const nor prior are given, the parameter is interpreted as being
        # defined as latent with an uninformative prior
        if const is None and prior is None:
            prior = Uninformative()

        # add the parameter to the central parameter dictionary
        if isinstance(prior, ProbabilityDistribution):  # i.e. adding 'latent'-parameter
            # first, define the index of this parameter in the numeric vector theta,
            # which is given to self.loglike and self.logprior
            prm_index = self.n_latent_prms_dim  # type: Union[int, None]
            prm_dim = dim
            prm_domain = domain
            assert dim is not None  # this check has been added for the mypy-check
            if count_intervals(domain) == 1 and dim > 1:
                prm_domain *= dim
            # the prm_value is reserved for 'const'-parameter; hence, it is set to None
            # in this case, where we are adding a 'latent'-param.
            prm_value = None
            # the remaining code in this if-branch defines the prior that is associated
            # with this 'latent'-parameter

            # extract the prior's elements
            prior_type = prior.dist_type  # e.g. 'normal', 'lognormal', etc.

            prior_dict = prior.prm_dict  # dictionary with parameter-value pairs
            prior_parameter_names = []  # type: List[Union[str, dict]]
            for prior_parameter_name, value in prior_dict.items():
                # create unique name for this prior parameter
                new_name = f"{prior_parameter_name}_{prm_name}"
                if type(value) in {float, int, list, tuple, np.ndarray}:
                    # in this case, the prior-parameter is considered a 'const'-
                    # parameter and added to the problem accordingly here
                    default_info = f"{prior_type.capitalize()} "
                    default_info += f"prior's parameter "
                    default_info += f"for latent parameter '{prm_name}'"
                    # the following call is recursive, but only with a depth of one,
                    # since the added parameter is a constant here
                    self.add_parameter(
                        new_name, "prior", const=value, info=default_info
                    )
                    prior_parameter_names.append(new_name)
                elif type(value) is str:
                    # in this case the prior-parameter is defined as an already defined
                    # parameter with the name stated in value
                    self.confirm_that_parameter_exists(value)
                    prior_parameter_names.append({value: new_name})
                else:
                    raise TypeError(
                        f"The prior-parameter {new_name} is not assigned a "
                        f"float, int or str, but something of type "
                        f"{type(value)}."
                    )
            prior_name = f"{prm_name}_{prior_type}"  # unique name of this prior
            prm_prior = PriorBase(
                prm_name, prior_parameter_names, prior_name, prior
            )  # type: Union[PriorBase, None]
            logger.debug(
                f"Adding  latent  {prm_type}-parameter "
                f"{prm_name} with {prior_type} prior to problem"
            )

        elif prior is None:
            # in this case we are adding a 'const'-parameter, which means that the
            # prm_index and prm_prior values are not used here
            prm_index = None
            prm_dim = len_or_one(const)
            prm_domain = None  # type: ignore
            prm_prior = None
            prm_value = convert_to_tuple(const)  # type: ignore
            logger.debug(
                f"Adding constant {prm_type}-parameter "
                f"{prm_name} = {prm_value} to problem"
            )
        else:
            raise TypeError(
                f"The 'prior'-flag of parameter '{prm_name}' is neither a "
                f"ProbabilityDistribution nor None."
            )

        # add the parameter to the central parameter dictionary
        self[prm_name] = ParameterProperties(
            {
                "index": prm_index,
                "dim": prm_dim,
                "domain": prm_domain,
                "type": prm_type,
                "prior": prm_prior,
                "value": prm_value,
                "info": info,
                "tex": tex,
            }
        )

    def __setitem__(self, key: str, value: "ParameterProperties"):
        """
        Performs some checks before adding a parameter to the dictionary.

        Parameters
        ----------
        key
            The key of the key-value pair to be added to self.
        value
            The value of the key-value pair to be added to self.
        """
        if type(key) != str:
            raise ValueError(
                f"The key must be a parameters name (string), but you provided "
                f"something of type '{type(key)}'."
            )
        if type(value) != ParameterProperties:
            raise ValueError(
                f"The properties of your parameter must be given in form of an "
                f"ParameterProperties-object. But you provided something of "
                f"type '{type(value)}'."
            )
        super().__setitem__(key, value)

    def __delitem__(self, key: str):
        """
        Deletes an item from itself while taking care of additional actions. For example
        removing prior-parameters when deleting a latent parameter or keeping the index-
        attributes of the latent parameters consistent.

        Parameters
        ----------
        key
            The key of the key-value pair to be removed from self.
        """

        # the given key is a parameter's name (renaming for easier readability)
        prm_name = key

        # check if the given parameter exists
        self.confirm_that_parameter_exists(prm_name)

        # different steps must be taken depending on whether the parameter which should
        # be removed is a 'const'- or a 'latent'-parameter
        if self[prm_name].index is None:
            # in this case prm_name refers to a constant parameter; hence, we can simply
            # remove this parameter without having to take care of other things as we
            # will have to do for latent parameters
            dict.__delitem__(self, key)
        else:
            # in this case prm_name refers to a latent parameter; hence we need to also
            # remove the prior-parameters; also, we have to correct the index values of
            # the remaining latent parameters
            for prior_prm in self[prm_name].prior.hyperparameters.keys():
                self.__delitem__(prior_prm)  # recursive call
            dict.__delitem__(self, prm_name)
            # correct the indices of the remaining 'latent'-parameters; note that the
            # way how the correction is done is due to the fact that the parameter.index
            # attribute is protected, and cannot be changed directly from outside
            idx_dict = {}
            idx = 0
            for prm_name, parameter in self.items():
                if parameter.is_latent:
                    dim = parameter.dim
                    idx_dict[prm_name] = idx
                    idx += dim
            for prm_name, idx in idx_dict.items():
                self[prm_name] = self[prm_name].changed_copy(index=idx)

    def confirm_that_parameter_exists(self, prm_name: str):
        """
        Checks if a parameter, given by its name, exists among the currently defined
        parameters. An error is raised when the given parameter does not exist yet.

        Parameters
        ----------
        prm_name
            A global parameter name.
        """
        if prm_name not in self:
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet."
            )

    def confirm_that_parameter_does_not_exists(self, prm_name: str):
        """
        Checks if a parameter, given by its name, exists among the currently defined
        parameters. An error is raised when the given parameter does already exist.

        Parameters
        ----------
        prm_name
            A global parameter name.
        """
        if prm_name in self:
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has already been defined."
            )

    @property
    def prms(self) -> List[str]:
        """Access the names of all parameters as an attribute."""
        return [*self.keys()]

    @property
    def n_prms(self) -> int:
        """Access the number of all parameters as an attribute."""
        return len(self)

    @property
    def latent_prms(self) -> List[str]:
        """Access the names of all 'latent'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.is_latent]

    @property
    def latent_prms_dims(self) -> List[int]:
        """Access the individual dimensions of the latent parameters."""
        return [self[prm_name].dim for prm_name in self.latent_prms]

    @property
    def n_latent_prms(self) -> int:
        """Access the number of all 'latent'-parameters as an attribute."""
        return len(self.latent_prms)

    @property
    def n_latent_prms_dim(self) -> int:
        """Access the combined dimension of all latent parameters. This number is the
        number of elements in the theta vector."""
        return sum(self.latent_prms_dims)

    @property
    def constant_prms(self) -> List[str]:
        """Access the names of all 'const'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.is_const]

    @property
    def constant_prms_dict(self) -> dict:
        """Access the names and values of all 'const'-param. as an attribute."""
        return {name: prm.value for name, prm in self.items() if prm.is_const}

    @property
    def n_constant_prms(self) -> int:
        """Access the number of all 'const'-parameters as an attribute."""
        return len(self.constant_prms)

    @property
    def model_prms(self) -> List[str]:
        """Access the names of all 'model'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.type == "model"]

    @property
    def n_model_prms(self) -> int:
        """Access the number of all 'model'-parameters as an attribute."""
        return len(self.model_prms)

    @property
    def prior_prms(self) -> List[str]:
        """Access the names of all 'prior'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.type == "prior"]

    @property
    def n_prior_prms(self) -> int:
        """Access the number of all 'prior'-parameters as an attribute."""
        return len(self.prior_prms)

    @property
    def likelihood_prms(self) -> List[str]:
        """Access the names of all 'likelihood'-parameters as an attribute."""
        return [name for name, prm in self.items() if prm.type == "likelihood"]

    @property
    def n_likelihood_prms(self) -> int:
        """Access the number of all 'likelihood'-parameters as an attribute."""
        return len(self.likelihood_prms)

    @property
    def value_dict(self) -> dict:
        """
        Returns a dict with the parameter names as keys and their numeric values as
        values. A parameter will only have a value if it is a constant. For latent
        parameters the dictionary-value will be None
        """
        return {name: props.value for name, props in self.items()}

    def parameter_overview(self, tablefmt: str = "presto") -> str:
        """
        Returns a string providing an overview of the defined parameters.

        Parameters
        ----------
        tablefmt
            An argument for the tabulate function defining the style of the generated
            table. Check out tabulate's documentation for more info.

        Returns
        -------
        prm_string
            This string describes a nice table with some essential information on the
            parameters of the problem.
        """
        # each element describes one row in the table to be generated
        rows = [
            (
                "Model parameters",
                simplified_list_string(self.model_prms),
                self.n_model_prms,
            ),
            (
                "Prior parameters",
                simplified_list_string(self.prior_prms),
                self.n_prior_prms,
            ),
            (
                "Likelihood parameters",
                simplified_list_string(self.likelihood_prms),
                self.n_likelihood_prms,
            ),
            (
                "Const parameters",
                simplified_list_string(self.constant_prms),
                self.n_constant_prms,
            ),
            (
                "Latent parameters",
                simplified_list_string(self.latent_prms),
                self.n_latent_prms,
            ),
        ]
        # these are the strings appearing in the column headers
        headers = ["Parameter type/role", "Parameter names", "Count"]
        prm_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prm_string = titled_table("Parameter overview", prm_table)
        return prm_string

    def parameter_explanations(self, tablefmt: str = "presto") -> str:
        """
        Returns a string providing short explanations on the defined parameters.

        Parameters
        ----------
        tablefmt
            An argument for the tabulate function defining the style of the generated
            table. Check out tabulate's documentation for more info.

        Returns
        -------
        prm_string
            This string describes a nice table with short explanations on the parameters
            of the problem.
        """
        rows = [(name, prm.info) for name, prm in self.items()]
        headers = ["Name", "Short explanation"]
        prm_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prm_string = titled_table("Parameter explanations", prm_table)
        return prm_string

    def const_parameter_values(self, tablefmt: str = "presto") -> str:
        """
        Returns a string providing the values of the defined 'const'-parameters.

        Parameters
        ----------
        tablefmt
            An argument for the tabulate function defining the style of the generated
            table. Check out tabulate's documentation for more info.

        Returns
        -------
        prm_string
            This string describes a nice table with the names and values of the constant
            parameters of the problem.
        """
        rows = [
            (name, prm.value) for name, prm in self.items() if prm.value is not None
        ]
        headers = ["Name", "Value"]
        prm_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prm_string = titled_table("Constant parameters", prm_table)
        return prm_string


class ParameterProperties:
    """
    Describes relevant properties of a ('latent' or 'const') parameter. Objects from
    this class are associated with the parameter's name in the dictionary class
    'Parameters', see above. The use of this class as opposed to a standard dictionary
    allows convenient auto-completion while coding.

    Parameters
    ----------
    prm_dict
        The keys are 'index', 'dim', 'type', 'role', 'prior', 'value', 'info' and 'tex',
        while the values are the corresponding values of these properties. See also the
        explanations in InverseProblem.__init__() for more detailed information.
    """

    def __init__(self, prm_dict: dict):

        # write attributes
        self._index = prm_dict["index"]
        self._type = prm_dict["type"]
        self._domain = prm_dict["domain"]
        self._prior = prm_dict["prior"]
        self._value = prm_dict["value"]
        self.info = prm_dict["info"]
        self.tex = prm_dict["tex"]

        # the dimension (dim) attribute is only expected to be contained in the given
        # prm_dict, if an index is specified; if no index is specified, a constant is
        # given, which defines its dimension based on its value
        if self._index is None:
            # constant parameter
            self._dim = len_or_one(self._value)
        else:
            # latent parameter
            self._dim = prm_dict["dim"]
            if type(prm_dict["domain"]) == str:
                self._domain = self.translate_domain_string(prm_dict["domain"])

            # check if each component has its domain
            if self._dim != len(self._domain):
                raise RuntimeError(
                    f"The dimension of the parameter ({self._dim}) is not identical "
                    f"with the number of provided domains ({len(self._domain)})!"
                )

        # whitespace in the tex strings is a problem for some plotting routines, so they
        # are replaced here by a math-command for whitespace that does not contain
        # actual whitespace
        if self.tex:
            self.tex = self.tex.replace(" ", r"$\enspace$")

        # check the given values
        self.check_consistency()

    @staticmethod
    def translate_domain_string(domain_string: str) -> list:
        """
        Translate a domain string like "(0, 1]" into a list of ScalarInterval objects.

        Parameters
        ----------
        domain_string
            A string like "(0, 1]" or "[0, 1] [0, 1]" defining the domain of a (possibly
            vector-valued) parameter.

        Returns
        -------
        intervals
            List of ScalarInterval objects derived form 'domain_string'.
        """

        # perform simple check on the given domain string
        _ = count_intervals(domain_string)

        # extract the intervals
        p_number = r"[-+]?[0-9]*\.?[0-9]*"
        p_infinity = r"[+-]?oo"
        p_value = rf"{p_number}|{p_infinity}"
        pattern = re.compile(rf"([\[(])({p_value})\W*?({p_value})([])])")
        interval_groups = pattern.findall(domain_string)
        intervals = []
        for interval_group in interval_groups:
            lower_bound_included = interval_group[0] == "["
            lower_bound = translate_number_string(interval_group[1])
            upper_bound = translate_number_string(interval_group[2])
            upper_bound_included = interval_group[3] == "]"
            intervals.append(
                ScalarInterval(
                    lower_bound, upper_bound, lower_bound_included, upper_bound_included
                )
            )

        return intervals

    # noinspection PyShadowingBuiltins
    def changed_copy(
        self,
        index: Optional[int] = None,
        dim: Optional[int] = None,
        domain: Union[tuple, List[tuple]] = None,
        type: Optional[str] = None,
        prior: Union[list, tuple, None] = None,
        value: Union[int, float, np.ndarray, None] = None,
        info: Optional[str] = None,
        tex: Optional[str] = None,
    ) -> "ParameterProperties":
        """
        Convenience method that simplifies changing the attributes of a
        ParameterProperties object based on creating a new instance. The reason for this
        approach is that some of the attributes are private, and cannot (or at least
        should not) be changed directly from outside.

        See the explanations in InverseProblem.__init__() for more detailed
        information on the arguments.
        """
        return ParameterProperties(
            {
                "index": index if index is not None else self._index,
                "dim": dim if dim is not None else self._dim,
                "domain": domain if domain is not None else self._domain,
                "type": type if type is not None else self._type,
                "prior": prior if prior is not None else self._prior,
                "value": value if value is not None else self._value,
                "info": info if info is not None else self.info,
                "tex": tex if tex is not None else self.tex,
            }
        )

    def check_consistency(self):
        """
        Checks the defined attributes in both isolated checks (each attribute is checked
        without considering others) and cross-checks, where the combination of
        attributes is checked on consistency.
        """

        # ------------------------------- #
        #         Isolated checks         #
        # ------------------------------- #

        if not (type(self._index) == int or self._index is None):
            raise TypeError(
                f"Found invalid ParameterProperties._index attribute! It must be of "
                f"type int or None, but found {type(self._index)}."
            )

        if (self._index is not None) and (self._index < 0):
            raise ValueError(
                f"Found negative value for ParameterProperties._index! This attribute "
                f"must be a non-negative integer, but found a value of {self._index}."
            )

        if not (type(self._dim) == int or self._dim is None):
            raise TypeError(
                f"Found invalid ParameterProperties._dim attribute! It must be of type "
                f"int or None, but found {type(self._dim)}."
            )

        if self._type not in ["model", "prior", "likelihood", "not defined"]:
            raise RuntimeError(
                f"Found invalid ParameterProperties._type attribute! It can only "
                f"assume the three values 'model', 'prior', 'likelihood' (or "
                f"'not defined' when automatic type-assignment is used) but found "
                f"'{self._type}'."
            )

        if not (isinstance(self._prior, PriorBase) or self._prior is None):
            raise TypeError(
                f"Found invalid ParameterProperties._prior attribute! It must be of "
                f"type PriorBase or None, but found {type(self._prior)}."
            )

        # -------------------------------- #
        #           Cross checks           #
        # -------------------------------- #

        if self._index is not None:
            # in this case, we have a latent parameter
            if self._value is not None:
                raise RuntimeError(
                    f"ParameterProperties._index and ParameterProperties._value are "
                    f"both given (_index={self._index} and _value={self._value}), but "
                    f"one of them must be None!"
                )
            if self._prior is None:
                raise RuntimeError(
                    f"When ParameterProperties._index is not None "
                    f"ParameterProperties._prior cannot be None!"
                )

        else:
            # in this case, we have a constant parameter
            if self._value is None:
                raise RuntimeError(
                    f"ParameterProperties._index and ParameterProperties._value are "
                    f"both None, but one of them must be not None!"
                )
            if self._prior is not None:
                raise RuntimeError(
                    f"ParameterProperties._index is None while Parameter"
                    f"Properties._prior is given ({self._prior}). This combination is "
                    f"not valid. Either the index must also be given, or the prior "
                    f"must also be None."
                )

    @property
    def index(self) -> int:
        """Access self._index from outside via self.index."""
        return self._index

    @index.setter
    def index(self, value: Union[int, float]):
        """Raise a specific error when trying to directly set self.index."""
        raise AttributeError("Changing a parameter's index directly is prohibited!")

    @property
    def dim(self) -> int:
        """Access self._dim from outside via self.dim."""
        return self._dim

    @dim.setter
    def dim(self, value: int):
        """Raise a specific error when trying to directly set self.dim."""
        raise AttributeError(
            "Changing a parameter's dimension (dim) directly is prohibited!"
        )

    @property
    def domain(self) -> Union[tuple, list]:
        """Access self._domain from outside via self.domain."""
        return self._domain

    @domain.setter
    def domain(self, value: Union[tuple, list]):
        """Raise a specific error when trying to directly set self.domain."""
        raise AttributeError("Changing a parameter's domain directly is prohibited!")

    @property
    def index_end(self) -> int:
        """Adds a pseudo-attribute self.index_end, which allows a convenient
        access to the (not-inclusive) end index in the parameter vector."""
        return self._index + self._dim

    @property
    def type(self) -> str:
        """Access self._type from outside via self.type."""
        return self._type

    @type.setter
    def type(self, value: str):
        """Raise a specific error when trying to directly set self.type."""
        raise AttributeError("Changing a parameter's type directly is prohibited!")

    @property
    def role(self) -> str:
        """Adds a pseudo-attribute self.role, which allows a convenient check
        on whether a parameter is latent or not."""
        return "latent" if self._index is not None else "const"

    @role.setter
    def role(self, value: str):
        """Raise a specific error when trying to directly set self.role."""
        raise AttributeError(
            "You cannot change a parameter's role directly! Use "
            "InverseProblem.change_parameter_role instead."
        )

    @property
    def is_latent(self) -> bool:
        """Adds a pseudo-attribute self.is_latent, which allows a convenient
        check on whether a parameter is latent or not."""
        return self._index is not None

    @property
    def is_const(self) -> bool:
        """Adds a pseudo-attribute self.is_const, which allows a convenient
        check on whether a parameter is constant or not."""
        return not self.is_latent

    @property
    def prior(self) -> Union[tuple, list, None]:
        """Access self._prior from outside via self.prior."""
        return self._prior

    @prior.setter
    def prior(self, value: Union[tuple, list, None]):
        """Raise a specific error when trying to directly set self.prior."""
        raise AttributeError("Changing a parameter's prior directly is prohibited!")

    @property
    def value(self) -> Union[int, float]:
        """Access self._value from outside via self.value."""
        return self._value

    @value.setter
    def value(self, value: Union[int, float]):
        """Raise a specific error when trying to directly set self.value."""
        raise AttributeError("Changing a parameter's value directly is prohibited!")


class ScalarInterval:
    """
    Describes a one-dimensional interval. Used for the domain-definition of parameters.

    Parameters
    ----------
    lower_bound
        The lower bound of the interval (if the interval is [a, b], this here is a).
    upper_bound
        The upper bound of the interval (if the interval is [a, b], this here is b).
    lower_bound_included
        Defines if the lower bound is included in the interval.
    upper_bound_included
        Defines if the upper bound is included in the interval.
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        lower_bound_included: bool,
        upper_bound_included: bool,
    ):
        # write arguments to attributes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_bound_included = lower_bound_included
        self.upper_bound_included = upper_bound_included

        # set the check_bounds-method; the distinction with respect to the inclusion of
        # bounds is done here and not in the check_bounds-method so that this check is
        # not repeated each time this method is called
        if lower_bound_included and upper_bound_included:
            self.check_bounds = self.check_bounds_inc_inc
        elif (not lower_bound_included) and upper_bound_included:
            self.check_bounds = self.check_bounds_ninc_inc
        elif lower_bound_included and (not upper_bound_included):
            self.check_bounds = self.check_bounds_inc_ninc
        else:
            self.check_bounds = self.check_bounds_ninc_ninc

    def check_bounds_inc_inc(self, value: Union[int, float]) -> bool:
        """
        Checks if a given value is within the specified bounds (where both bounds are
        included).

        Parameters
        ----------
        value
            The given scalar value.

        Returns
        -------
            True, if the value is within its bounds, otherwise False is returned.
        """
        if self.lower_bound <= value <= self.upper_bound:
            return True
        else:
            return False

    def check_bounds_ninc_inc(self, value: Union[int, float]) -> bool:
        """
        Checks if a given value is within the specified bounds (where only the upper
        bound is included).

        Parameters
        ----------
        value
            The given scalar value.

        Returns
        -------
            True, if the value is within its bounds, otherwise False is returned.
        """
        if self.lower_bound < value <= self.upper_bound:
            return True
        else:
            return False

    def check_bounds_inc_ninc(self, value: Union[int, float]) -> bool:
        """
        Checks if a given value is within the specified bounds (where only the lower
        bound is included).

        Parameters
        ----------
        value
            The given scalar value.

        Returns
        -------
            True, if the value is within its bounds, otherwise False is returned.
        """
        if self.lower_bound <= value < self.upper_bound:
            return True
        else:
            return False

    def check_bounds_ninc_ninc(self, value: Union[int, float]) -> bool:
        """
        Checks if a given value is within the specified bounds (where only the upper
        bound is included).

        Parameters
        ----------
        value
            The given scalar value.

        Returns
        -------
            True, if the value is within its bounds, otherwise False is returned.
        """
        if self.lower_bound < value < self.upper_bound:
            return True
        else:
            return False

    def __str__(self):
        s1 = "[" if self.lower_bound_included else "("
        s2 = "-oo" if self.lower_bound == -np.infty else self.lower_bound
        s3 = "+oo" if self.upper_bound == np.infty else self.upper_bound
        s4 = "]" if self.upper_bound_included else ")"
        return f"{s1}{s2}, {s3}{s4}"
