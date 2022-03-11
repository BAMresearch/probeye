# standard library
import copy
from typing import Union, List, Optional, Callable
import copy as cp
import urllib
import os

# third party imports
from tabulate import tabulate
from loguru import logger
import numpy as np
import torch as th

# local imports
from probeye.definition.parameter import Parameters
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.subroutines import underlined_string, titled_table
from probeye.subroutines import simplified_list_string
from probeye.subroutines import make_list, len_or_one
from probeye.subroutines import print_probeye_header
from probeye.subroutines import logging_setup
from probeye.subroutines import add_index_to_tex_prm_name
from probeye.subroutines import translate_simple_correlation


class InferenceProblem:
    """
    This class provides a general framework for defining an inference problem (more
    specifically, a parameter estimation problem) without specifying or providing any
    computational means for solving the problem.

    Parameters
    ----------
    name
        This is the name of the problem and has only descriptive value, for example when
        working with several inference problems.
    use_default_logger
        When True, the logger will be set up with some useful default values. Otherwise,
        no logger configurations are applied and a logger can be defined outside of the
        problem definition.
    log_level
        The log-level used by the default logger for printing to std out. This argument
        is intended for quickly controlling the amount of logging output the user sees
        on the screen when running probeye.
    log_file
        Path to the log-file, if the logging-stream should be printed to file. If None
        is given, no logging-file will be created.
    print_header
        If True, a probeye header is logged when an instance of this class is created.
        Otherwise, the header will not be logged.
    """

    def __init__(
        self,
        name: str,
        use_default_logger: bool = True,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        print_header: bool = True,
    ):

        # the name of the problem
        self.name = name
        self.use_default_logger = use_default_logger
        self.log_level = log_level
        self.log_file = log_file
        self.print_header = print_header

        # this is the central parameter dictionary of the problem (the used Parameters-
        # class is derived from the dict-class); it contains all defined parameters
        # ('const' and 'latent' ones); the keys of this dictionary are the parameter
        # names (note that each parameter must have a unique name in the problem); the
        # values of this dictionary are Parameter-objects (see parameter.py) with the
        # following attributes:
        # .index  int or None (the index of a 'latent' parameter in the latent parameter
        #         vector theta; None for 'const'-parameters)
        # .dim    int or None (the dimension of the parameter if it is latent, in case
        #         of a constant parameter, this attribute is None)
        # .type   string (either 'model', 'prior' or 'likelihood' depending on where the
        #         parameter appears)
        # .prior  object or None (the prior-object of the 'latent'-parameter; None for
        #         'const'-parameters)
        # .value  float or None (defines the value for 'const'-parameters; None for
        #         'latent'-parameters)
        # .info   string (a short explanation of the parameter)
        # .tex:   string or None (the TeX version of the parameter's name, for example
        #         r'$\alpha$' for a parameter named 'alpha'; only used for plotting)
        # this attribute is managed internally and should not be modified directly
        self._parameters = Parameters()

        # this dictionary is intended for storing the problem's experimentally measured
        # data; this dict is managed internally and should not be modified directly
        self._experiments = {}  # type: dict

        # the following dict contains the problem's forward models; note that a single
        # problem can have multiple forward models; the keys in this dict are the
        # forward model names, while the values are the forward model objects (check out
        # the script forward_model.py); this dictionary is managed internally and should
        # not be modified directly
        self._forward_models = {}  # type: dict

        # this dictionary contains the problem's likelihood models; as the other private
        # attributes above, it is managed internally and should not be modified directly
        self._likelihood_models = {}  # type: dict

        # setup the logger with the given specifications
        if use_default_logger:
            logging_setup(log_file=log_file, log_level_stdout=log_level)

        # log probeye header and first message
        if print_header:
            print_probeye_header()
            logger.debug("")  # for visual separation
        logger.debug(f"Initialized inference problem: '{self.name}'")

    @property
    def n_prms(self) -> int:
        """Provides n_prms attribute."""
        return self._parameters.n_prms

    @property
    def prms(self) -> List[str]:
        """Provides prms attribute."""
        return self._parameters.prms

    @property
    def n_latent_prms(self) -> int:
        """Provides n_latent_prms attribute."""
        return self._parameters.n_latent_prms

    @property
    def n_latent_prms_dim(self) -> int:
        """Provides n_latent_prms_dim attribute."""
        return self._parameters.n_latent_prms_dim

    @property
    def latent_prms(self) -> List[str]:
        """Provides latent_prms attribute."""
        return self._parameters.latent_prms

    @property
    def latent_prms_dims(self) -> List[int]:
        """Provides latent_prms_dims attribute."""
        return self._parameters.latent_prms_dims

    @property
    def n_constant_prms(self) -> int:
        """Provides n_constant_prms attribute."""
        return self._parameters.n_constant_prms

    @property
    def constant_prms(self) -> List[str]:
        """Provides constant_prms attribute."""
        return self._parameters.constant_prms

    @property
    def constant_prms_dict(self) -> dict:
        """Provides constant_prms_dict attribute."""
        return self._parameters.constant_prms_dict

    @property
    def n_model_prms(self) -> int:
        """Provides n_model_prms attribute."""
        return self._parameters.n_model_prms

    @property
    def model_prms(self) -> List[str]:
        """Provides model_prms attribute."""
        return self._parameters.model_prms

    @property
    def n_prior_prms(self) -> int:
        """Provides n_prior_prms attribute."""
        return self._parameters.n_prior_prms

    @property
    def prior_prms(self) -> List[str]:
        """Provides prior_prms attribute."""
        return self._parameters.prior_prms

    @property
    def n_likelihood_prms(self) -> int:
        """Provides n_likelihood_prms attribute."""
        return self._parameters.n_likelihood_prms

    @property
    def likelihood_prms(self) -> List[str]:
        """Provides likelihood_prms attribute."""
        return self._parameters.likelihood_prms

    @property
    def parameters(self) -> Parameters:
        """Access self._parameters from outside via self.parameters."""
        return self._parameters

    @property
    def priors(self) -> dict:
        """Provides the problem's prior-dictionary which is derived dynamically from
        the latent parameters in the self.parameters dictionary. The keys are the
        priors names, while the values are the prior-objects."""
        return {
            parameter_properties.prior.name: parameter_properties.prior
            for parameter_properties in self.parameters.values()
            if parameter_properties.is_latent
        }

    @property
    def likelihood_models(self) -> dict:
        """Access self._likelihood_models from outside via self.likelihood_models."""
        return self._likelihood_models

    @property
    def forward_models(self) -> dict:
        """Access self._forward_models from outside via self.forward_models."""
        return self._forward_models

    @property
    def experiments(self) -> dict:
        """Access self._experiments from outside via self.experiments."""
        return self._experiments

    # =============================================================== #
    #                    Parameter related methods                    #
    # =============================================================== #

    def add_parameter(
        self,
        prm_name: str,
        prm_type: str,
        dim: Optional[int] = 1,
        domain: str = "(-oo, +oo)",
        const: Union[int, float, np.ndarray, None] = None,
        prior: Union[tuple, list, None] = None,
        info: str = "No explanation provided",
        tex: Optional[str] = None,
    ):
        """
        Adds a parameter ('const' or 'latent') to the inference problem. For more
        information, check out the Parameters.add_parameter method.
        """

        # add the parameter to the central parameter dictionary; checks and translations
        # are conducted by the Parameters.add_parameter method
        self._parameters.add_parameter(
            prm_name,
            prm_type,
            dim=dim,
            domain=domain,
            const=const,
            prior=prior,
            info=info,
            tex=tex,
        )

    def remove_parameter(self, prm_name: str):
        """
        Removes a parameter ('const' or 'latent') from inference problem.

        Parameters
        ----------
        prm_name
            The name of the parameter to be removed.
        """

        # checks/additional actions are conducted by Parameters' __delitem__ method
        del self._parameters[prm_name]

    def change_parameter_role(
        self,
        prm_name: str,
        const: Union[int, float, None] = None,
        prior: Union[tuple, None] = None,
        domain: Union[tuple, List[tuple]] = (-np.infty, np.infty),
    ):
        """
        Performs the necessary tasks to change a parameter's role in the problem
        definition. A parameter's role can either be changed from 'const' to 'latent'
        or from 'latent' to 'const'. Note that parameter roles cannot be modified in
        any other way.

        Parameters
        ----------
        prm_name
            The name of the parameter whose role should be changed.
        const
            If the new role is 'const', the corresponding value has to be specified by
            this argument.
        prior
            If the new role is 'latent', this argument has to be given as a 2-tuple.
            Check out the explanations in self.add_parameter for more information on
            this argument.
        domain
            The parameter's domain (i.e., values it may assume). Note that this argument
            is only considered for latent parameter, but not for a constant.
        """
        # first, make sure that the given parameter exists
        self._parameters.confirm_that_parameter_exists(prm_name)

        # exactly one of the const and prior key word arguments must be given
        if const is not None and prior is not None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key argument. You "
                f"have specified both."
            )
        if const is None and prior is None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key argument. You "
                f"have specified none."
            )
        # raise an error if the role change would not change the role
        if self._parameters[prm_name].is_const and (prior is None):
            raise RuntimeError(
                f"The parameter '{prm_name}' is already defined as constant."
            )
        if self._parameters[prm_name].is_latent and (const is None):
            raise RuntimeError(
                f"The parameter '{prm_name}' is already defined as a "
                f"latent parameter."
            )
        # the parameter's role is changed by first removing it from the problem, and
        # then adding it again in its new role; the role-change does not impact the type
        # ('model', 'prior' or 'likelihood')
        prm = self._parameters[prm_name]
        if prm.is_const:
            # if a constant parameter should be made latent, its dimension will be taken
            # from its current value
            dim = len_or_one(prm.value)  # type: Union[int, None]
        else:
            # in this case, a latent parameter should be made constant; since the dim-
            # attribute will be inferred from the given value (const) it does not need
            # to be specified here
            dim = None
        self.remove_parameter(prm_name)
        self.add_parameter(
            prm_name,
            prm.type,
            dim=dim,
            domain=domain,
            const=const,
            prior=prior,
            info=prm.info,
            tex=prm.tex,
        )

    def change_parameter_type(self, prm_name: str, new_type: str):
        """
        Changes the type of a parameter, i.e., 'model', 'prior' or 'likelihood'. Note
        that parameter types cannot be modified in any other way.

        Parameters
        ----------
        prm_name
            The name of the parameter whose type should be changed.
        new_type
            The new type, either 'model', 'prior' or 'likelihood'.
        """
        # first, make sure that the given parameter exists
        self._parameters.confirm_that_parameter_exists(prm_name)

        # make sure, the requested type is valid
        if new_type not in ["model", "prior", "likelihood"]:
            raise ValueError(
                f"Requested type change of parameter '{prm_name}' to the invalid type "
                f"'{new_type}'. Valid options are 'model', 'prior' or 'likelihood'."
            )

        # change the type as requested
        self._parameters[prm_name] = self._parameters[prm_name].changed_copy(
            type=new_type
        )

    def change_parameter_info(
        self,
        prm_name: str,
        new_info: Optional[str] = None,
        new_tex: Optional[str] = None,
    ):
        """
        Changes the info-string and/or the tex-string of a given parameter. This task
        can also be done directly since the 'info' and 'tex' attributes of the
        parameters are not protected. However, to do so, one needs to know some of the
        internal structure. With this function, one does not need to know anything about
        the internals. So, it is more a convenience function (which however might not be
        needed very often).

        Parameters
        ----------
        prm_name
            The name of the parameter whose info/tex-string should be changed.
        new_info
            The new string for the explanation of parameter prm_name.
        new_tex
            The new string for the parameter's tex-representation.
        """
        # first, make sure that the given parameter exists
        self._parameters.confirm_that_parameter_exists(prm_name)

        # if None is given for the new info/tex, the old value will be kept
        if new_info is None:
            new_info = self._parameters[prm_name].info
        if new_tex is None:
            new_tex = self._parameters[prm_name].tex

        # change the info/tex-string
        self._parameters[prm_name] = self._parameters[prm_name].changed_copy(
            info=new_info, tex=new_tex
        )

    def change_constant(self, prm_name: str, new_value: Union[int, float]):
        """
        Changes the value of a 'const'-parameter, i.e. a constant parameter of the
        inference problem. Note that constants cannot be modified in any other way.

        Parameters
        ----------
        prm_name
            The name of the 'const'-parameter whose value should be changed.
        new_value
            The new value that prm_name should assume.
        """
        # first, make sure that the given parameter exists
        self._parameters.confirm_that_parameter_exists(prm_name)

        # check if the given parameter is a constant
        if self._parameters[prm_name].is_latent:
            raise RuntimeError(f"The parameter '{prm_name}' is not a constant!")
        # change the parameter's value
        self._parameters[prm_name] = self._parameters[prm_name].changed_copy(
            value=new_value
        )

    def get_parameters(
        self, theta: Union[np.ndarray, th.Tensor], prm_def: dict
    ) -> dict:
        """
        Extracts the numeric values for given parameters that have been defined within
        the inference problem. The numeric values are extracted either from the latent
        parameter vector theta or from the constant parameters of the problem.

        Parameters
        ----------
        theta
            A numeric parameter vector passed to the loglike and logprior method. Which
            parameters these numbers refer to can be checked by calling the method
            self.theta_explanation() once the problem is set up.
        prm_def
            Defines which parameters to extract. The keys of this dictionary are the
            global parameter names, while the values are the local parameter names. In
            most cases global and local names will be identical, but sometimes it is
            convenient to define a local parameter name, e.g. in the forward model.

        Returns
        -------
        prms
            Contains <local parameter name> : <(global) parameter value> pairs. If a
            parameter is scalar, its value will be returned as a float. In case of a
            vector-valued parameter, its value will be returned either as a np.ndarray
            or a th.Tensor, depending on the format theta was provided in.
        """
        prms = {}
        for global_name, local_name in prm_def.items():
            idx = self._parameters[global_name].index
            if idx is None:
                # in this case, the parameter is a constant and hence not read from
                # theta, but from the internal library
                prms[local_name] = self._parameters[global_name].value
            else:
                # in this case, the parameter is a latent parameter, and its value is
                # read from theta
                idx_end = self._parameters[global_name].index_end
                if idx == (idx_end - 1):
                    # scalars should not be returned as one-element-lists
                    prms[local_name] = theta[idx]
                else:
                    prms[local_name] = theta[idx:idx_end]
        return prms

    def get_theta_names(self, tex: bool = False, components: bool = False) -> list:
        """
        Returns the parameter names of the latent parameter vector theta in the
        corresponding order (either in tex- or conventional format).

        Parameters
        ----------
        tex
            If True, the TeX-names of the parameters will be returned, otherwise the
            global names as they are used in the code will be returned.
        components
            If True, parameters with dimension > 1 are returned component-wise; e.g.,
            if 'alpha' is a 2D parameter, it will be returned as 'alpha_1' and
            'alpha_2'. If False, only 'alpha' would be returned.

        Returns
        -------
        theta_names
            List of strings with the parameter names appearing in theta.
        """
        # assemble the parameter's names in the order as they appear in theta
        theta_names = []
        indices = []
        for prm_name, parameter in self._parameters.items():
            if parameter.is_latent:
                # add the parameter indices for checking later
                idx = parameter.index
                dim = parameter.dim
                if components and (dim > 1):
                    idx_end = parameter.index_end
                    indices += [*range(idx, idx_end)]
                else:
                    indices.append(parameter.index)
                # add the parameter names
                if tex and parameter.tex is not None:
                    if components and (dim > 1):
                        for i in range(dim):
                            theta_names.append(
                                add_index_to_tex_prm_name(parameter.tex, i + 1)
                            )
                    else:
                        theta_names.append(parameter.tex)
                else:
                    if components and (dim > 1):
                        for i in range(dim):
                            theta_names.append(f"{prm_name}_{i + 1}")
                    else:
                        theta_names.append(prm_name)
        # order the theta_names according to their index-values; note that this step is
        # not necessary for insertion ordered dicts (Python 3.6+), since in this case
        # theta_names will already be in the right order
        theta_names = [name for _, name in sorted(zip(indices, theta_names))]
        return theta_names

    def theta_explanation(self, check_consistency: bool = True) -> str:
        """
        Returns a string describing how the theta-vector, which is the numeric latent
        parameter vector that is given to the likelihood and prior methods, is
        interpreted with respect to the problem's parameters. The printout will tell
        you which parameter is connected to which index of theta.

        Parameters
        ----------
        check_consistency
            When True, a consistency check is performed before printing the
            explanations on theta. When False, this check is skipped.

        Returns
        -------
        s
            The constructed string.
        """

        # an explanation is not printed if the problem is inconsistent
        if check_consistency:
            self.check_problem_consistency()

        # collect the list of theta names in the right order
        theta_names = self.get_theta_names()

        # construct the info-string
        theta_string = "\n+---------------------------+\n"
        theta_string += "|  Theta  |    Parameter    |\n"
        theta_string += "|  index  |      name       |\n"
        theta_string += "|---------------------------|\n"
        for prm_name in theta_names:
            idx = self.parameters[prm_name].index
            dim = self.parameters[prm_name].dim
            if dim == 1:
                theta_string += f"|{idx:7d} --> {prm_name:<15s}|\n"
            else:
                idx_str = f"{idx}-{idx + dim - 1}"
                theta_string += f"|{idx_str:>7s} --> {prm_name:<15s}|\n"
        theta_string += "+---------------------------+\n"

        return theta_string

    # =============================================================== #
    #                  Forward model related methods                  #
    # =============================================================== #

    def add_forward_model(self, name: str, forward_model: ForwardModelBase):
        """
        Adds a forward model to the inference problem. Note that multiple forward models
        can be added to one problem.

        Parameters
        ----------
        name
            The name of the forward model to be added.
        forward_model
            Defines the forward model. Check out forward_model.py to see a template for
            the forward model definition. The user will then have to derive his own
            forward model from that base class. Examples can be found in the package
            directory tests/integration_tests.
        """

        # log at beginning so that errors can be quickly associated
        logger.debug(f"Adding forward model '{name}'")

        # check if all given model parameters have already been added to the inference
        # problem; note that the forward model can only be added to the problem after
        # the corresponding parameters have been defined
        for prm_name in forward_model.prms_def:
            self._parameters.confirm_that_parameter_exists(prm_name)

        # check if the given name for the forward model has already been used
        if name in self._forward_models:
            raise RuntimeError(
                f"The given name '{name}' for the forward model has already been used "
                f"for another forward model. Please choose another name."
            )

        # check if the given forward model has an output sensor with a name that is
        # already used for an output sensor of another forward model
        for existing_name, existing_fwd_model in self._forward_models.items():
            for output_sensor in existing_fwd_model.output_sensor_names:
                if output_sensor in forward_model.output_sensor_names:
                    raise RuntimeError(
                        f"The given forward model '{name}' has an output sensor "
                        f"'{output_sensor}', \nwhich is also defined as an output "
                        f"sensor in the already defined forward model "
                        f"'{existing_name}'.\nPlease choose a different name for "
                        f"output sensor '{output_sensor}' in forward model '{name}'."
                    )

        # add the given forward model to the internal forward model dictionary under
        # the given forward model name
        self._forward_models[name] = forward_model

    # =============================================================== #
    #                   Experiments related methods                   #
    # =============================================================== #

    def add_experiment(
        self,
        exp_name: str,
        sensor_values: dict,
        fwd_model_name: str,
        correlation_info: Union[str, dict, None] = None,
    ):
        """
        Adds a single experiment to the inference problem. An experiment is simply a
        collection of measured data which was produced by one event. The measured data
        is given in form of a dictionary (sensor_values). Additionally, an experiment
        must contain a reference to one and only one forward model (fwd_model_name).
        This should be understood as follows: The experiment contains data that is (at
        least in part) modeled by the stated forward model.

        Parameters
        ----------
        exp_name
            The name of the experiment, e.g. "Exp_20May.12". If an experiment with a
            similar name has already been added, it will be overwritten and a warning
            will be thrown.
        sensor_values
            The keys are the sensor's names (for example 'strain_gauge_1') and the
            values are the measured values.
        fwd_model_name
            Name of the forward model this experiment refers to.
        correlation_info
            If no correlation should be considered between the data of the experiment,
            this argument must be None. However, if there is correlated data this
            argument must be a dictionary where the keys are sensor names of the
            experiment and the value are dictionaries. These dictionaries have as keys
            strings that describe correlation variables and as values again sensor names
            of the experiment. To given an example: {"y1": {"x": "x1", "t": "t"},
            "y2": {"x": "x2", "t": "t"}}. Here, y1, x1, y2, x2, and t must appear as
            sensor names in the experiment, and it means that y1 and y2 are correlated
            in x and t where the corresponding x data of y1/y2 is x1/x2 and the
            corresponding t data of y1/y2 is simply t (no alias). In simple cases with
            only one correlation variable that has a standard name, for example
            {"y1": {"x": "x"}}, a string can be given. In the example given before, this
            string would be "y1:x".
        """

        # log at beginning so that errors can be associated
        logger.debug(f"Adding experiment '{exp_name}'")

        # make sure that no likelihood model has been defined yet
        if self._likelihood_models:
            raise RuntimeError(
                f"You are trying to add an experiment after already having defined "
                f"likelihood models.\nSince version 1.0.17 this is not allowed "
                f"anymore. Please add all experiments before\n defining any likelihood "
                f"model. After you have defined a likelihood model, you cannot add\n"
                f"experiments anymore."
            )

        # check types of input arguments
        if type(sensor_values) is not dict:
            raise TypeError(
                f"The sensor_values must be given as a dictionary. However, "
                f"found type '{type(sensor_values)}'."
            )
        if type(fwd_model_name) is not str:
            raise TypeError(
                f"The fwd_model_name must be given as a string. However, "
                f"found type '{type(fwd_model_name)}'."
            )
        for sensor_name, sensor_value in sensor_values.items():
            if type(sensor_value) not in [list, tuple, np.ndarray, float, int]:
                raise ValueError(
                    f"The sensor value of sensor '{sensor_name}' stated for experiment "
                    f"'{exp_name}' is of type '{type(sensor_value)}'. However, "
                    f"only 'list', 'tuple', 'np.ndarray', 'float' or 'int' are allowed."
                )

        # check if the given forward model exists
        if fwd_model_name not in self._forward_models.keys():
            raise RuntimeError(
                f"The forward model '{fwd_model_name}' does not exist! You need to "
                f"define it before adding experiments that refer to it."
            )
        fwd_model = self._forward_models[fwd_model_name]

        # check that the stated forward model is consistent with the experiment; to be
        # consistent, the experiment must provide data to all input sensors of the
        # referenced forward model and must also contain data to the forward model's
        # output sensor (otherwise the forward model's response could not be compared
        # with the experimental data)
        experiment_sensors = [*sensor_values.keys()]
        input_sensors = fwd_model.input_sensors
        for input_sensor in input_sensors:
            if input_sensor.name not in experiment_sensors:
                raise RuntimeError(
                    f"The forward model's ({fwd_model_name}) input sensor "
                    f"'{input_sensor.name}' is not provided by the given "
                    f"experiment '{exp_name}'!"
                )
        output_sensors = fwd_model.output_sensors
        for output_sensor in output_sensors:
            if output_sensor.name not in experiment_sensors:
                raise RuntimeError(
                    f"The forward model's ({fwd_model_name}) output sensor "
                    f"'{output_sensor.name}' is not provided by the given "
                    f"experiment '{exp_name}'!"
                )

        # check that the sensor values are given in the right format; only floats, ints,
        # np.ndarray, lists and tuples are allowed; lists and tuples will be converted
        # to 1D np.ndarrays; multidimensional np.ndarrays will lead to a value error
        sensor_values_numpy = cp.deepcopy(sensor_values)
        for sensor_name, values in sensor_values.items():
            if isinstance(values, (list, tuple)):
                sensor_values_numpy[sensor_name] = np.atleast_1d(np.array(values))
            elif isinstance(values, np.ndarray):
                if values.ndim > 1:
                    raise ValueError(
                        f"The sensor values of an experiment must be given as 1D "
                        f"arrays. However, the sensor_values of '{sensor_name}' in "
                        f"experiment '{exp_name}' are given as an {values.ndim}D array."
                    )
                if values.size == 1:
                    raise ValueError(
                        f"Encountered a np.ndarray with only one element in "
                        f"experiment '{exp_name}', sensor '{sensor_name}'. "
                        f"Please use floats or integers to provide scalar data."
                    )
                sensor_values_numpy[sensor_name] = values
            elif isinstance(values, (float, int)):
                sensor_values_numpy[sensor_name] = values
            else:
                raise ValueError(
                    f"Encountered invalid type '{type(values)}' in the sensor values "
                    f"of sensor '{sensor_name}' in experiment '{exp_name}'.\n Sensor"
                    f"data must be given as int, float, np.ndarray, list or tuple."
                )

        # throw warning when the experiment name was defined before
        if exp_name in self._experiments.keys():
            logger.warning(
                f"An experiment with name '{exp_name}' is already defined within the "
                f"problem scope.\nIt will now be overwritten! Please make sure that "
                f"this was intended."
            )

        # check the given correlation information
        if correlation_info is not None:
            # for simple cases the correlation_info can be given as a string
            if isinstance(correlation_info, str):
                correlation_info = translate_simple_correlation(correlation_info)
            for key_sensor, corr_dict in correlation_info.items():
                # the keys of correlation_info must be sensor names of the experiment
                if key_sensor not in sensor_values:
                    raise RuntimeError(
                        f"The key '{key_sensor}' in the argument correlation_info does"
                        f" not appear in the given sensor_values!"
                    )
                # they keys of correlation_info must be names of output sensors of the
                # experiment's forward model
                if key_sensor not in fwd_model.output_sensor_names:
                    raise RuntimeError(
                        f"The correlation_info of experiment '{exp_name}' states the "
                        f"sensor '{key_sensor}' as a key without this sensor\nbeing "
                        f"the name of an output sensor of forward model "
                        f"'{fwd_model_name}' (this is the forward model this "
                        f"experiment refers to)."
                    )
                # the values must be dicts again
                if not isinstance(corr_dict, dict):
                    raise TypeError(
                        f"Found invalid format of the 'correlation_info' argument in "
                        f"experiment '{exp_name}'!\nThe required format is a 2-level "
                        f"dictionary. However, the value of key '{key_sensor}' is not "
                        f"a dictionary but a '{type(corr_dict)}'."
                    )
                # also the last-level values must be sensor names of the experiment
                for val_sensor in corr_dict.values():
                    if val_sensor not in sensor_values:
                        raise RuntimeError(
                            f"The value '{val_sensor}' in the argument correlation_info"
                            f" does not appear in the given sensor_values!"
                        )

        # add the experiment to the central dictionary
        self._experiments[exp_name] = {
            "sensor_values": sensor_values_numpy,
            "forward_model": fwd_model_name,
            "correlation_info": correlation_info,
        }

    def get_experiment_names(
        self,
        forward_model_names: Union[str, List[str], None] = None,
        sensor_names: Union[str, List[str], None] = None,
        experiment_names: Union[str, List[str], None] = None,
    ) -> list:
        """
        Extracts the names of all experiments which refer to a forward model from a
        given list and/or which contain a set of given sensors. The experiments are
        extracted from a given set of experiments.

        Parameters
        ----------
        forward_model_names
            The names of the forward model the experiments should refer to. This means,
            to be selected, an experiment must refer to one of those fwd model names.
        sensor_names
            The names of the sensors the experiments should should contain. To be
            selected, an experiment must contain all of the sensors stated in this list.
        experiment_names
            The names of the experiments to sub-select from. If None is given, then all
            experiments of the problem will be used.

        Returns
        -------
        relevant_experiment_names
            The names of the sub-selected experiments.
        """

        # at least one of forward_model_names and sensor_names must be given
        if (not forward_model_names) and (not sensor_names):
            raise RuntimeError(
                f"You did not specify any forward model(s) or sensor names."
            )

        # if experiments is not further specified it is assumed that all given
        # experiments should be used
        if experiment_names is None:
            experiment_names = [*self._experiments.keys()]

        # this is for collecting the experiments
        relevant_experiment_names = []

        # get the experiments which refer to one of the given forward models; keep in
        # mind that each experiment refers to exactly one forward model
        if forward_model_names is not None:
            forward_model_names_list = make_list(forward_model_names)
            for exp_name in experiment_names:
                fwd_model_name = self._experiments[exp_name]["forward_model"]
                if fwd_model_name in forward_model_names_list:
                    relevant_experiment_names.append(exp_name)
            experiment_names = relevant_experiment_names

        # get the experiments from the previously sub-selected experiments which
        # contain all of the given sensors
        if sensor_names is not None:
            relevant_experiment_names = []
            sensor_names_list = make_list(sensor_names)
            for exp_name in experiment_names:
                exp_sensors = [*self._experiments[exp_name]["sensor_values"].keys()]
                if all([s in exp_sensors for s in sensor_names_list]):
                    relevant_experiment_names.append(exp_name)

        return relevant_experiment_names

    def transform_experimental_data(
        self, func: Callable, args: tuple = (), **kwargs
    ) -> "InferenceProblem":
        """
        Returns a copy of the problem the experimental data of which is transformed in
        some way. This might be a necessary pre-processing step for an inference engine
        in order to be able to solve the problem. Note that the problem is not fully
        deep-copied. The forward models are excluded from deep-copying, as this might
        result in problems. The rest of the problem is deep-copied however.

        Parameters
        ----------
        func
            The function that is applied on each of the experiment's sensor values.
        args
            Additional positional arguments to be passed to func.
        kwargs
            Keyword arguments to be passed to func.

        Returns
        -------
        self_copy
            A copy of self where the experimental data has been transformed in the
            specified fashion.
        """

        # for easier error tracing
        logger.debug(f"Transforming experimental data using f = '{func.__name__}'")

        # the original problem shall not be touched, so we create a copy here to which
        # the transformation will be applied
        self_copy = InferenceProblem(
            self.name,
            use_default_logger=self.use_default_logger,
            log_level=self.log_level,
            log_file=self.log_file,
            print_header=False,
        )
        self_copy._parameters = cp.deepcopy(self._parameters)
        self_copy._experiments = cp.deepcopy(self._experiments)
        self_copy._forward_models = cp.copy(self._forward_models)  # no deep-copy here!
        self_copy._likelihood_models = cp.deepcopy(self._likelihood_models)
        # the following step is necessary since the attribute 'problem_experiments' of
        # the deep-copied likelihood models in self_copy still refer to the experiments
        # of self; hence, we need to set this pointer to the experiments of self_copy
        for likelihood_model in self_copy._likelihood_models.values():
            likelihood_model.problem_experiments = self_copy._experiments

        # transform the sensor values from the experiments by applying the specified
        # function with the given arguments to them
        for exp_name in self_copy._experiments.keys():
            sensor_values = self_copy._experiments[exp_name]["sensor_values"]
            for sensor_name in sensor_values.keys():
                sensor_values[sensor_name] = func(
                    sensor_values[sensor_name], *args, **kwargs
                )

        return self_copy

    # ================================================================ #
    #                 Likelihood model related methods                 #
    # ================================================================ #

    def add_likelihood_model(
        self, likelihood_model: GaussianLikelihoodModel, name: Optional[str] = None
    ):
        """
        Adds a likelihood model to the inference problem. Note that a single problem
        can have multiple likelihood models. It is assumed that all likelihood models
        of a problem are mutually independent. Note that before adding a likelihood
        model, all the experiments the likelihood model should refer to must have
        already been added to the InferenceProblem.

        Parameters
        ----------
        likelihood_model
            The general likelihood model object, see likelihood.py. This likelihood
            model is general in that sense, that it is merely descriptive without
            providing any computational means. The general likelihood model will be
            translated into a likelihood model with computational means in the solver
            that will later be used to 'solve' the inference problem.
        name
            A descriptive name for the likelihood model. If None is given, a default
            name will be derived automatically (something like 'likelihood_model_1').
        """

        # check if the likelihood model has been assigned a name; if not, assign one
        if name is None:
            name = f"likelihood_model_{len(self.likelihood_models)}"
            logger.debug(
                f"Adding likelihood model '{name}' (name assigned automatically)"
            )
        else:
            logger.debug(f"Adding likelihood model '{name}'")
        likelihood_model.name = name  # this is important e.g. for the torch-solver!

        # ensure that experiments have already been added to the problem
        if not self._experiments:
            raise RuntimeError(
                f"You are trying to add a likelihood model to your problem without "
                f"having added any experiments yet!\nPlease add the experiments "
                f"before adding the likelihood models."
            )

        # check if all given likelihood model parameters have already been added to
        # the inference problem
        for prm_name in likelihood_model.prms_def:
            if prm_name not in self._parameters.keys():
                raise RuntimeError(
                    f"The likelihood model parameter '{prm_name}' has not been defined "
                    f"yet.\nYou have to add all likelihood model parameters to the "
                    f"problem before adding the likelihood model.\nYou can use the"
                    f"'add_parameter' method for this purpose."
                )

        # add the problem's experiments to the likelihood model (note that this is just
        # a pointer!) for likelihood_model-internal checks
        likelihood_model.problem_experiments = self._experiments

        # check/assign the likelihood model's experiments
        if len(likelihood_model.experiment_names) == 0:
            # in this case, the likelihood model will be assigned its experiments
            # automatically; this assignment works via the likelihood model's sensors
            # (at least when the sensors have been specified by the user); it is simply
            # checked which experiments contain all of the likelihood model's sensors as
            # sensor values; those experiments will be assigned then
            logger.debug(
                f"No experiments were explicitly defined for likelihood model "
                f"'{likelihood_model.name}'."
            )
            logger.debug(
                f"The following experiments were added were added automatically "
                f"to {name}':"
            )
            if len(likelihood_model.sensors) > 0:
                added_experiment_names = self.get_experiment_names(
                    sensor_names=likelihood_model.sensor_names
                )
                likelihood_model.add_experiments(added_experiment_names)
                for exp_name in added_experiment_names:
                    logger.debug(f"{likelihood_model.name} <--- {exp_name}")
            else:
                # in this case, the user did not specify the likelihood model's sensors
                # and also did not specify the assigned experiments; this minimal
                # specification is interpreted as all of the problem's experiments being
                # assigned to the likelihood model
                for exp_name in self._experiments:
                    likelihood_model.add_experiments(exp_name)
                    logger.debug(f"{likelihood_model.name} <--- {exp_name}")

        # set the likelihood's forward model
        likelihood_model.determine_forward_model()

        # the following case is relevant when the user did not specify the likelihood
        # model's sensors when initializing a GaussianLikelihoodModel instance; this
        # case is mostly for convenience
        if len(likelihood_model.sensors) == 0:
            logger.debug(
                f"No sensors were assigned to likelihood model "
                f"'{likelihood_model.name}'."
            )
            logger.debug(f"Assigning sensors automatically based on its forward model.")
            likelihood_model.sensors = self.forward_models[
                likelihood_model.forward_model
            ].output_sensors
            logger.debug(f"Assigned sensors: {likelihood_model.sensor_names}")

        # this step is necessary here again, since the likelihood model's
        # add_experiments-method does a few things more, when the likelihood model's
        # sensors are defined; this is given for sure not before this point
        experiment_names_user = copy.copy(likelihood_model.experiment_names)
        likelihood_model.experiment_names = []
        likelihood_model.add_experiments(experiment_names_user)

        # finally, add the likelihood_model to the internal dict
        self._likelihood_models[name] = likelihood_model

    # =============================================================== #
    #                          Other methods                          #
    # =============================================================== #

    def info(
        self,
        tablefmt: str = "presto",
        check_consistency: bool = True,
        print_header: bool = False,
        return_string: bool = False,
    ) -> Union[str, None]:
        """
        Logs an overview of the problem definition and returns the generated string.

        Parameters
        ----------
        tablefmt
            An argument for the tabulate function defining the style of the generated
            table. Check out tabulate's documentation for more info.
        check_consistency
            When True, a consistency check is performed before printing the explanations
            on theta. When False, this check is skipped.
        print_header
            When True, the probeye header is printed before printing the problem
            information. Otherwise, the header is not printed.
        return_string
            When True, the constructed string is returned. Otherwise it is just logged
            without it being returned.

        Returns
        -------
            The constructed string providing the problem's definition if 'return_string'
            was set to True. Otherwise, None is returned.
        """

        # print the header if requested
        if print_header:
            print_probeye_header()

        # state the name of the inference problem
        title = f"Problem summary: {self.name}"
        title_string = underlined_string(title, n_empty_start=1)

        # provide information on the defined forward models
        rows_fwd_model = []
        for name, model in self._forward_models.items():
            glob_prms = simplified_list_string([*model.prms_def.keys()])
            loc_prms = simplified_list_string([*model.prms_def.values()])
            rows_fwd_model.append((name, glob_prms, loc_prms))
        headers = ["Model name", "Global parameters", "Local parameters"]
        fwd_table = tabulate(rows_fwd_model, headers=headers, tablefmt=tablefmt)
        fwd_string = titled_table("Forward models", fwd_table)

        # provide information on the defined priors
        rows_prior = []
        for name, prior in self.priors.items():
            glob_prms = simplified_list_string([*prior.prms_def.keys()])
            loc_prms = simplified_list_string([*prior.prms_def.values()])
            rows_prior.append((name, glob_prms, loc_prms))
        headers = ["Prior name", "Global parameters", "Local parameters"]
        prior_table = tabulate(rows_prior, headers=headers, tablefmt=tablefmt)
        prior_str = titled_table("Priors", prior_table)

        # provide various information on the problem's parameters
        prm_string = self._parameters.parameter_overview(tablefmt=tablefmt)
        prm_string += self._parameters.parameter_explanations(tablefmt=tablefmt)
        prm_string += self._parameters.const_parameter_values(tablefmt=tablefmt)

        # include the information on the theta interpretation
        theta_string = "\nTheta interpretation"
        theta_string += self.theta_explanation(check_consistency=check_consistency)

        # provide information on the added experiments
        max_length = 0
        max_digits = 0
        for exp_dict in self._experiments.values():
            for sensor_name, sensor_values in exp_dict["sensor_values"].items():
                if len(sensor_name) > max_length:
                    max_length = len(sensor_name)
                if len(str(len_or_one(sensor_values))) > max_digits:
                    max_digits = len(str(len_or_one(sensor_values)))
        rows_exp = []  # type: List[tuple[str, str, str]]
        for exp_name, exp_dict in self._experiments.items():
            sensor_name_1 = [*exp_dict["sensor_values"].keys()][0]
            sensor_values_1 = [*exp_dict["sensor_values"].values()][0]
            n_values_1 = len_or_one(sensor_values_1)
            sv_info_1 = f"{sensor_name_1:{max_length}} "
            if n_values_1 == 1:
                sv_info_1 += f"({n_values_1:{max_digits}} element)"
            else:
                sv_info_1 += f"({n_values_1:{max_digits}} elements)"
            rows_exp.append((exp_name, sv_info_1, exp_dict["forward_model"]))
            # the remaining code is for experiments with more than one sensor val
            if len([*exp_dict["sensor_values"].keys()]) > 1:
                for sensor_name, sensor_values in exp_dict["sensor_values"].items():
                    if sensor_name == sensor_name_1:
                        continue
                    n_values = len_or_one(sensor_values)
                    sv_info = f"{sensor_name:{max_length}} "
                    if n_values == 1:
                        sv_info += f"({n_values:{max_digits}} element)"
                    else:
                        sv_info += f"({n_values:{max_digits}} elements)"
                    rows_exp.append(("", sv_info, ""))
        headers = ["Name", "Sensor values", "Forward model"]
        exp_table = tabulate(rows_exp, headers=headers, tablefmt=tablefmt)
        exp_str = titled_table("Added experiments", exp_table)

        # provide information on the added likelihood models
        rows_like = []
        for l_name, l_model in self._likelihood_models.items():
            prms_glob = simplified_list_string([*l_model.prms_def.keys()])
            prms_loc = simplified_list_string([*l_model.prms_def.values()])
            l_sensors = simplified_list_string(l_model.sensor_names)
            exp_name_1 = l_model.experiment_names[0]
            rows_like.append((l_name, prms_glob, prms_loc, l_sensors, exp_name_1))
            for exp_name in l_model.experiment_names:
                if exp_name == exp_name_1:
                    continue
                rows_like.append(("", "", "", "", exp_name))
        headers = ["Name", "Glob. prms", "Loc. prms", "Target sensors", "Experiments"]
        like_table = tabulate(rows_like, headers=headers, tablefmt=tablefmt)
        like_str = titled_table("Added likelihood models", like_table)

        # concatenate the string and return it
        full_string = title_string + fwd_string + prior_str + prm_string
        full_string += theta_string + exp_str + like_str

        # log and return the string
        if return_string:
            return full_string
        else:
            for line in full_string.split("\n"):
                logger.info(line)
            return None

    def __str__(self) -> str:
        """
        Allows to print the problem definition via print(problem) if problem is
        an instance of InferenceProblem. See self.info for more details.
        """
        return self.info(return_string=True)  # type: ignore

    def check_problem_consistency(self):
        """
        Conducts various checks to make sure the problem definition does not contain
        any inconsistencies.
        """

        # check if the central components have been added to the problem: parameters,
        # forward models, experiments and likelihood models
        assert (
            len(self._parameters) != 0
        ), "The problem does not contain any parameters!"
        assert (
            len(self._forward_models) != 0
        ), "The problem does not contain any forward models!"
        assert (
            len(self._experiments) != 0
        ), "The problem does not contain any experiments!"
        assert (
            len(self._likelihood_models) != 0
        ), "The problem does not contain any likelihood models!"

        # check if all parameters of the forward model(s) appear in self._parameters
        # and if they have the correct type
        for fwd_name, forward_model in self._forward_models.items():
            for model_prm in forward_model.prms_def:
                assert model_prm in self._parameters, (
                    f"The forward model parameter '{model_prm}' that appears in "
                    f"forward model '{fwd_name}' is not defined within the problem!"
                )
                assert self._parameters[model_prm].type == "model", (
                    f"The forward model parameter '{model_prm}' that appears in "
                    f"forward model '{fwd_name}' has not been defined as of type "
                    f"'model' but of type '{self._parameters[model_prm].type}' in the "
                    f"problem definition!"
                )

        # check if all parameters of the likelihood model appear in self._parameters
        # and if they have the correct type
        for likelihood_name, likelihood_model in self._likelihood_models.items():
            for likelihood_prm in likelihood_model.prms_def:
                assert likelihood_prm in self._parameters, (
                    f"The likelihood model parameter '{likelihood_prm}' that appears "
                    f"in likelihood model '{likelihood_name}' is not defined within "
                    f"the problem!"
                )
                assert self._parameters[likelihood_prm].type == "likelihood", (
                    f"The likelihood model parameter '{likelihood_prm}' that appears "
                    f"in likelihood model '{likelihood_name}' has not been defined "
                    f"as of type 'likelihood' but of type "
                    f"'{self._parameters[likelihood_prm].type}' in the problem "
                    f"definition!"
                )

        # check if all prior objects in self.priors are consistent in terms of their
        # parameters; each one of them must appear in self._parameters
        for prior_name, prior_obj in self.priors.items():
            for prior_prm in prior_obj.hyperparameters:
                assert prior_prm in self._parameters, (
                    f"The prior parameter '{prior_prm}' that appears in prior "
                    f"'{prior_name}' is not defined within the problem!"
                )
                assert self._parameters[prior_prm].type == "prior", (
                    f"The prior parameter '{prior_prm}' that appears in prior "
                    f"'{prior_name}' has not been defined as of type 'prior' but of "
                    f"type '{self._parameters[prior_prm].type}' in the problem "
                    f"definition!"
                )

        # check the indices of the latent parameters; the combined indices should form
        # an ascending vector of integers from 0 to n; no multiple or missing values
        # are allowed
        idx_list = []
        for prm_name, parameter in self._parameters.items():
            if parameter.is_latent:
                idx_range = [*range(parameter.index, parameter.index_end)]
                idx_list += idx_range
        assert sorted(idx_list) == list(range(len(idx_list))), (
            f"There seems to be an inconsistency in the latent parameter's indices. "
            f"The sorted index list is: {sorted(idx_list)}"
        )

        # check the consistency of each parameter
        for parameter in self._parameters.values():
            parameter.check_consistency()

        # check the consistency of each likelihood model
        for likelihood_model in self._likelihood_models.values():
            likelihood_model.check_experiment_consistency()

        # check that each defined experiment appears in one of the likelihood models
        exp_names_in_likelihood_models = set()
        for likelihood_model in self._likelihood_models.values():
            for exp_name_likelihood in likelihood_model.experiment_names:
                exp_names_in_likelihood_models.add(exp_name_likelihood)
        for exp_name in self._experiments.keys():
            if exp_name not in exp_names_in_likelihood_models:
                logger.warning(
                    f"The globally defined experiment '{exp_name}' does not appear in "
                    f"any of the likelihood models!"
                )
