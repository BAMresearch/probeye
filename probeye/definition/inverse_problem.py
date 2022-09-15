# standard library
from typing import Union, List, Optional, Tuple

# third party imports
from tabulate import tabulate
from loguru import logger
import numpy as np

# local imports
from probeye.definition.parameter import Parameters
from probeye.definition.experiment import Experiment
from probeye.definition.distribution import ProbabilityDistribution
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.subroutines import underlined_string, titled_table
from probeye.subroutines import simplified_list_string
from probeye.subroutines import make_list, len_or_one
from probeye.subroutines import print_probeye_header
from probeye.subroutines import logging_setup
from probeye.subroutines import add_index_to_tex_prm_name
from probeye.subroutines import safe_string
from probeye.subroutines import translate_prms_def


class InverseProblem:
    """
    This class provides a general framework for defining an inverse problem (more
    specifically, a parameter estimation problem) without specifying or providing any
    computational means for solving the problem.

    Parameters
    ----------
    name
        This is the name of the problem and has only descriptive value, for example when
        working with several inverse problems.
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

        # the name and safe name of the problem; the latter is intended for file naming
        self.name = name
        self.safe_name = safe_string(name)

        # logging related preparations
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
        self.parameters = Parameters()

        # this dict is intended for storing the problem's experimentally measured data
        self.experiments = {}  # type: dict

        # the following dict contains the problem's forward models; note that a single
        # problem can have multiple forward models; the keys in this dict are the
        # forward model names, while the values are the forward model objects (check out
        # the script forward_model.py)
        self.forward_models = {}  # type: dict

        # this dictionary contains the problem's likelihood models; as the other private
        # attributes above, it is managed internally and should not be modified directly
        self.likelihood_models = {}  # type: dict

        # setup the logger with the given specifications
        if use_default_logger:
            logging_setup(log_file=log_file, log_level_stdout=log_level)

        # log probeye header and first message
        if print_header:
            print_probeye_header()
            logger.debug("")  # for visual separation
        logger.debug(f"Initialized inverse problem: '{self.name}'")

    @property
    def n_prms(self) -> int:
        """Provides n_prms attribute."""
        return self.parameters.n_prms

    @property
    def prms(self) -> List[str]:
        """Provides prms attribute."""
        return self.parameters.prms

    @property
    def n_latent_prms(self) -> int:
        """Provides n_latent_prms attribute."""
        return self.parameters.n_latent_prms

    @property
    def n_latent_prms_dim(self) -> int:
        """Provides n_latent_prms_dim attribute."""
        return self.parameters.n_latent_prms_dim

    @property
    def latent_prms(self) -> List[str]:
        """Provides latent_prms attribute."""
        return self.parameters.latent_prms

    @property
    def latent_prms_dims(self) -> List[int]:
        """Provides latent_prms_dims attribute."""
        return self.parameters.latent_prms_dims

    @property
    def n_constant_prms(self) -> int:
        """Provides n_constant_prms attribute."""
        return self.parameters.n_constant_prms

    @property
    def constant_prms(self) -> List[str]:
        """Provides constant_prms attribute."""
        return self.parameters.constant_prms

    @property
    def constant_prms_dict(self) -> dict:
        """Provides constant_prms_dict attribute."""
        return self.parameters.constant_prms_dict

    @property
    def n_model_prms(self) -> int:
        """Provides n_model_prms attribute."""
        return self.parameters.n_model_prms

    @property
    def model_prms(self) -> List[str]:
        """Provides model_prms attribute."""
        return self.parameters.model_prms

    @property
    def n_prior_prms(self) -> int:
        """Provides n_prior_prms attribute."""
        return self.parameters.n_prior_prms

    @property
    def prior_prms(self) -> List[str]:
        """Provides prior_prms attribute."""
        return self.parameters.prior_prms

    @property
    def n_likelihood_prms(self) -> int:
        """Provides n_likelihood_prms attribute."""
        return self.parameters.n_likelihood_prms

    @property
    def likelihood_prms(self) -> List[str]:
        """Provides likelihood_prms attribute."""
        return self.parameters.likelihood_prms

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

    # =============================================================== #
    #                    Parameter related methods                    #
    # =============================================================== #

    def add_parameter(
        self,
        name: str,
        prm_type: str = "not defined",
        dim: Optional[int] = 1,
        domain: str = "(-oo, +oo)",
        value: Union[int, float, np.ndarray, None] = None,
        prior: Optional[ProbabilityDistribution] = None,
        info: str = "No explanation provided",
        tex: Optional[str] = None,
    ):
        """
        Adds a parameter ('const' or 'latent') to the inverse problem. For more
        information, check out the Parameters.add_parameter method.
        """

        # add the parameter to the central parameter dictionary; checks and translations
        # are conducted by the Parameters.add_parameter method
        self.parameters.add_parameter(
            name,
            prm_type,
            dim=dim,
            domain=domain,
            value=value,
            prior=prior,
            info=info,
            tex=tex,
        )

    def remove_parameter(self, prm_name: str):
        """
        Removes a parameter ('const' or 'latent') from inverse problem.

        Parameters
        ----------
        prm_name
            The name of the parameter to be removed.
        """

        # checks/additional actions are conducted by Parameters' __delitem__ method
        del self.parameters[prm_name]

    def change_parameter_role(
        self,
        prm_name: str,
        value: Union[int, float, None] = None,
        prior: Optional[ProbabilityDistribution] = None,
        domain: str = "(-oo, +oo)",
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
        value
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
        self.parameters.confirm_that_parameter_exists(prm_name)

        # exactly one of the const and prior key word arguments must be given
        if value is not None and prior is not None:
            raise RuntimeError(
                f"You must specify either the 'value' or the 'prior' key argument. You "
                f"have specified both."
            )
        if value is None and prior is None:
            raise RuntimeError(
                f"You must specify either the 'value' or the 'prior' key argument. You "
                f"have specified none."
            )
        # raise an error if the role change would not change the role
        if self.parameters[prm_name].is_const and (prior is None):
            raise RuntimeError(
                f"The parameter '{prm_name}' is already defined as constant."
            )
        if self.parameters[prm_name].is_latent and (value is None):
            raise RuntimeError(
                f"The parameter '{prm_name}' is already defined as a "
                f"latent parameter."
            )
        # the parameter's role is changed by first removing it from the problem, and
        # then adding it again in its new role; the role-change does not impact the type
        # ('model', 'prior' or 'likelihood')
        prm = self.parameters[prm_name]
        if prm.is_const:
            # if a constant parameter should be made latent, its dimension will be taken
            # from its current value
            dim = len_or_one(prm.value)  # type: Union[int, None]
        else:
            # in this case, a latent parameter should be made constant; since the dim-
            # attribute will be inferred from the given value (value) it does not need
            # to be specified here
            dim = None
        self.remove_parameter(prm_name)
        self.add_parameter(
            prm_name,
            prm.type,
            dim=dim,
            domain=domain,
            value=value,
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
        self.parameters.confirm_that_parameter_exists(prm_name)

        # make sure, the requested type is valid
        if new_type not in ["model", "prior", "likelihood"]:
            raise ValueError(
                f"Requested type change of parameter '{prm_name}' to the invalid type "
                f"'{new_type}'. Valid options are 'model', 'prior' or 'likelihood'."
            )

        # change the type as requested
        self.parameters[prm_name] = self.parameters[prm_name].changed_copy(
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
        self.parameters.confirm_that_parameter_exists(prm_name)

        # if None is given for the new info/tex, the old value will be kept
        if new_info is None:
            new_info = self.parameters[prm_name].info
        if new_tex is None:
            new_tex = self.parameters[prm_name].tex

        # change the info/tex-string
        self.parameters[prm_name] = self.parameters[prm_name].changed_copy(
            info=new_info, tex=new_tex
        )

    def change_constant(self, prm_name: str, new_value: Union[int, float]):
        """
        Changes the value of a 'const'-parameter, i.e. a constant parameter of the
        inverse problem. Note that constants cannot be modified in any other way.

        Parameters
        ----------
        prm_name
            The name of the 'const'-parameter whose value should be changed.
        new_value
            The new value that prm_name should assume.
        """
        # first, make sure that the given parameter exists
        self.parameters.confirm_that_parameter_exists(prm_name)

        # check if the given parameter is a constant
        if self.parameters[prm_name].is_latent:
            raise RuntimeError(f"The parameter '{prm_name}' is not a constant!")
        # change the parameter's value
        self.parameters[prm_name] = self.parameters[prm_name].changed_copy(
            value=new_value
        )

    def get_latent_prior_hyperparameters(self, prm_name: str) -> list:
        """
        Returns a list of the latent hyperparameters of a parameter's prior. In most
        cases there will be none, so an empty list will be returned.

        Parameters
        ----------
        prm_name
            The name of the parameter the prior of which should be checked for latent
            hyperparameters.

        Returns
        -------
        latent_hyperparameters
            Contains the global names of latent hyperparameters in the prior of the
            parameter 'prm_name'.
        """

        # first, make sure that the given parameter exists
        self._parameters.confirm_that_parameter_exists(prm_name)

        # now look for possible latent hyperparameters
        latent_hyperparameters = []
        if self.parameters[prm_name].is_latent:
            hyperparameters = self.parameters[prm_name].prior.hyperparameters
            for hyperparameter in hyperparameters:
                if self.parameters[hyperparameter].is_latent:
                    latent_hyperparameters.append(hyperparameter)

        return latent_hyperparameters

    def get_parameters(self, theta: np.ndarray, prm_def: dict) -> dict:
        """
        Extracts the numeric values for given parameters that have been defined within
        the inverse problem. The numeric values are extracted either from the latent
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
            vector-valued parameter, its value will be returned as a np.ndarray.
        """
        prms = {}
        for global_name, local_name in prm_def.items():
            idx = self.parameters[global_name].index
            if idx is None:
                # in this case, the parameter is a constant and hence not read from
                # theta, but from the internal library
                prms[local_name] = self.parameters[global_name].value
            else:
                # in this case, the parameter is a latent parameter, and its value is
                # read from theta
                idx_end = self.parameters[global_name].index_end
                if idx == (idx_end - 1):
                    # scalars should not be returned as one-element-lists
                    prms[local_name] = theta[idx]
                else:
                    prms[local_name] = theta[idx:idx_end]
        return prms

    def get_constants(self, prm_def: dict) -> dict:
        """
        Similar to 'get_parameters' with the difference that this method only extracts
        the numeric values of constants that have been defined within the problem scope.
        For that reason it does not need the 'theta' argument as in 'get_parameters'. If
        'prm_def' also contains latent parameters, they will simply be ignored.

        Parameters
        ----------
        prm_def
            Defines which constants to extract. The keys of this dictionary are the
            global parameter names, while the values are the local parameter names. In
            most cases global and local names will be identical, but sometimes it is
            convenient to define a local parameter name, e.g. in the forward model.

        Returns
        -------
        prms
            Contains <local parameter name> : <(global) parameter value> pairs. If a
            parameter is scalar, its value will be returned as a float. In case of a
            vector-valued parameter, its value will be returned as a np.ndarray.
        """
        prms = {}
        for global_name, local_name in prm_def.items():
            idx = self._parameters[global_name].index
            if idx is None:
                prms[local_name] = self._parameters[global_name].value
        return prms

    def check_parameter_domains(self, theta: np.ndarray) -> bool:
        """
        Checks whether the given values of the latent parameters are within their
        specified domains.

        Parameters
        ----------
        theta
            A numeric vector or tensor, which contains the current values of all latent
            parameters.

        Returns
        -------
            True if all values given by theta are within their specified domains.
            Otherwise, False is returned.
        """
        for theta_name in self.get_theta_names():
            theta_dim = self.parameters[theta_name].dim
            theta_idx = self.parameters[theta_name].index
            for i in range(theta_dim):
                domain = self.parameters[theta_name].domain[i]
                if not domain.check_bounds(float(theta[theta_idx + i])):
                    return False
        return True

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
        for prm_name, parameter in self.parameters.items():
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

    def add_forward_model(
        self, forward_model: ForwardModelBase, experiments: Union[str, list]
    ):
        """
        Adds a forward model to the inverse problem. Note that multiple forward models
        can be added to one problem.

        Parameters
        ----------
        forward_model
            Defines the forward model. Check out forward_model.py to see a template for
            the forward model definition. The user will then have to derive his own
            forward model from that base class. Examples can be found in the package
            directory tests/integration_tests.
        experiments
            A single string or a list of strings that represents the names of the
            problem's experiments that are described by forward_model.
        """

        # log at beginning so that errors can be quickly associated
        logger.debug(f"Adding forward model '{forward_model.name}'")

        # check if all given model parameters have already been added to the inverse
        # problem; note that the forward model can only be added to the problem after
        # the corresponding parameters have been defined
        for prm_name in forward_model.prms_def:
            self.parameters.confirm_that_parameter_exists(prm_name)
            # check, if the type was set correctly; if it was not set yet, it will be
            # set automatically here
            if self.parameters[prm_name].type == "not defined":
                self.change_parameter_type(prm_name, "model")
            elif self.parameters[prm_name].type in ["prior", "likelihood"]:
                raise ValueError(
                    f"The parameter '{prm_name}' defined in forward model "
                    f"'{forward_model.name}' was assigned the type "
                    f"'{self.parameters[prm_name].type}' (it should be 'model')."
                )

        # check if the given name for the forward model has already been used
        if forward_model.name in self.forward_models:
            raise RuntimeError(
                f"The name '{forward_model.name}' of the forward model you are trying "
                f"to add to the problem has already been used for another forward "
                f"model within the problem scope. Please choose another name."
            )

        # assign the given experiments to the forward model
        experiment_list = make_list(experiments)
        for exp_name in experiment_list:
            if exp_name not in self.experiments:
                raise RuntimeError(
                    f"The experiment '{exp_name}' that is supposed to be modeled by "
                    f"forward model '{forward_model.name}' has not been defined yet!"
                )
            for sensor_name in forward_model.sensor_names:
                if sensor_name not in self.experiments[exp_name].sensor_data:
                    raise RuntimeError(
                        f"The forward model's sensor '{sensor_name}' was not found in "
                        f"experiment '{exp_name}' that was assigned to it."
                    )
            self.experiments[exp_name].forward_model = forward_model.name

        # at this point, the forward model is stripped of its response method, since
        # the problem merely contains descriptive information about the inverse problem
        class ForwardModelHull(forward_model.__class__):  # type: ignore
            def response(self, inp):
                raise NotImplementedError(
                    "You requested a response from the forward model's hull, which "
                    "does not contain a 'response'-method! The forward model has to be "
                    "translated first in order to be equipped with a computing method."
                )

        # add an instance of the forward model's hull (which contains its 'interface'-
        # method but not its 'response'-method) to the problem
        fwd_model_hull = ForwardModelHull(
            forward_model.name, *forward_model.args, **forward_model.kwargs
        )
        fwd_model_hull.experiment_names = experiment_list
        self.forward_models[forward_model.name] = fwd_model_hull

    # =============================================================== #
    #                   Experiments related methods                   #
    # =============================================================== #

    def add_experiment(
        self,
        name: str,
        sensor_data: dict,
    ):
        """
        Adds a single experiment to the inverse problem. An experiment is simply a
        collection of measured data which was produced by one event. The measured data
        is given in form of a dictionary (sensor_data).

        Parameters
        ----------
        name
            The name of the experiment, e.g. "Exp_20May.12". If an experiment with a
            similar name has already been added, it will be overwritten and a warning
            will be thrown.
        sensor_data
            The keys are the sensor's names (for example 'strain_gauge_1') and the
            values are the measured values.
        """

        # log at beginning so that errors can be associated
        logger.debug(f"Adding experiment '{name}'")

        # create an experiment-object which will trigger some checks
        experiment = Experiment(name, sensor_data)

        # throw warning when the experiment name was defined before
        if name in self.experiments.keys():
            logger.warning(
                f"An experiment with name '{name}' is already defined within the "
                f"problem scope.\nIt will now be overwritten! Please make sure that "
                f"this was intended."
            )

        # add the experiment to the central dictionary
        self.experiments[name] = experiment

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
            experiment_names = [*self.experiments.keys()]

        # this is for collecting the experiments
        relevant_experiment_names = []

        # get the experiments which refer to one of the given forward models; keep in
        # mind that each experiment refers to exactly one forward model
        if forward_model_names is not None:
            forward_model_names_list = make_list(forward_model_names)
            for exp_name in experiment_names:
                fwd_model_name = self.experiments[exp_name].forward_model
                if fwd_model_name in forward_model_names_list:
                    relevant_experiment_names.append(exp_name)
            experiment_names = relevant_experiment_names

        # get the experiments from the previously sub-selected experiments which
        # contain all of the given sensors
        if sensor_names is not None:
            relevant_experiment_names = []
            sensor_names_list = make_list(sensor_names)
            for exp_name in experiment_names:
                exp_sensors = [*self.experiments[exp_name].sensor_data.keys()]
                if all([s in exp_sensors for s in sensor_names_list]):
                    relevant_experiment_names.append(exp_name)

        return relevant_experiment_names

    # ================================================================ #
    #                 Likelihood model related methods                 #
    # ================================================================ #

    def add_likelihood_model(
        self,
        likelihood_model: GaussianLikelihoodModel,
    ):
        """
        Adds a likelihood model to the inverse problem. Note that a single problem
        can have multiple likelihood models. It is assumed that all likelihood models
        of a problem are mutually independent. Note that before adding a likelihood
        model, all the experiments the likelihood model should refer to must have
        already been added to the InverseProblem.

        Parameters
        ----------
        likelihood_model
            The general likelihood model object, see likelihood.py. This likelihood
            model is general in that sense, that it is merely descriptive without
            providing any computational means. The general likelihood model will be
            translated into a likelihood model with computational means in the solver
            that will later be used to 'solve' the inverse problem.
        """

        # the likelihood models are added to the problem after all parameters have been
        # added; this is why this method includes a silent automatic parameter type
        # detection, to avoid an extra command in the problem definition
        for prm_name in self.parameters:
            if self.parameters[prm_name].type == "not defined":
                for prior in self.priors.values():
                    if prm_name in prior.hyperparameters:
                        self.change_parameter_type(prm_name, "prior")

        # ensure the likelihood model's experiment has been added to the problem
        if likelihood_model.experiment_name not in self.experiments:
            raise RuntimeError(
                f"The likelihood model's experiment {likelihood_model.experiment_name} "
                f"was not found in the problem's experiments."
            )

        # set the likelihood's forward model based on the likelihood's experiment; only
        # after the forward model has been set, the correlation definitions can be
        # checked on consistency which is done via process_correlation_definition()
        exp_name = likelihood_model.experiment_name
        fwd_model_name = self.experiments[exp_name].forward_model
        likelihood_model.forward_model = self.forward_models[fwd_model_name]

        # derive the likelihood model's parameters
        prms = []
        if likelihood_model.measurement_error is not None:
            prm_name = likelihood_model.measurement_error
            self.parameters.confirm_that_parameter_exists(prm_name)
            prms.append(prm_name)
            for output_sensor in likelihood_model.forward_model.output_sensors:
                output_sensor.std_measurement = prm_name
        if likelihood_model.correlation_model is not None:
            for prm_name in likelihood_model.correlation_model.parameters:
                self.parameters.confirm_that_parameter_exists(prm_name)
            prms += likelihood_model.correlation_model.parameters
        for osensor in likelihood_model.forward_model.output_sensors:
            prm_name = osensor.std_model
            self.parameters.confirm_that_parameter_exists(prm_name)
            prms.append(prm_name)
        prms_def, prms_dim = translate_prms_def(prms)
        likelihood_model.prms_def = prms_def
        likelihood_model.prms_dim = prms_dim

        # check if all given model parameters have already been added to the inverse
        # problem; note that the likelihood model can only be added to the problem after
        # the corresponding parameters have been defined
        for prm_name in likelihood_model.prms_def:
            self.parameters.confirm_that_parameter_exists(prm_name)
            # check, if the type was set correctly; if it was not set yet, it will be
            # set automatically here
            if self.parameters[prm_name].type == "not defined":
                self.change_parameter_type(prm_name, "likelihood")

        # finally, add the likelihood_model to the internal dict
        self.likelihood_models[likelihood_model.experiment_name] = likelihood_model

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

        # state the name of the inverse problem
        title = f"Problem summary: {self.name}"
        title_string = underlined_string(title, n_empty_start=1)

        # provide information on the defined forward models
        rows_fwd_model = []
        for name, model in self.forward_models.items():
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
        prm_string = self.parameters.overview(tablefmt=tablefmt)
        prm_string += self.parameters.parameter_explanations(tablefmt=tablefmt)
        prm_string += self.parameters.const_parameter_values(tablefmt=tablefmt)

        # include the information on the theta interpretation
        theta_string = "\nTheta interpretation"
        theta_string += self.theta_explanation(check_consistency=check_consistency)

        # provide information on the added experiments
        max_length = 0
        max_digits = 0
        for experiment in self.experiments.values():
            for sensor_name, sensor_data in experiment.sensor_data.items():
                if len(sensor_name) > max_length:
                    max_length = len(sensor_name)
                if len(str(len_or_one(sensor_data))) > max_digits:
                    max_digits = len(str(len_or_one(sensor_data)))
        rows_exp = []  # type: List[Tuple[str, str, str]]
        for exp_name, experiment in self.experiments.items():
            sensor_name_1 = experiment.sensor_names[0]
            sensor_data_1 = experiment.sensor_data[sensor_name_1]
            n_values_1 = len_or_one(sensor_data_1)
            sv_info_1 = f"{sensor_name_1:{max_length}} "
            if n_values_1 == 1:
                sv_info_1 += f"({n_values_1:{max_digits}} element)"
            else:
                sv_info_1 += f"({n_values_1:{max_digits}} elements)"
            rows_exp.append((exp_name, sv_info_1, experiment.forward_model))
            # the remaining code is for experiments with more than one sensor val
            if experiment.n_sensors > 1:
                for sensor_name, sensor_data in experiment.sensor_data.items():
                    if sensor_name == sensor_name_1:
                        continue
                    n_values = len_or_one(sensor_data)
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
        for l_name, l_model in self.likelihood_models.items():
            prms_glob = simplified_list_string([*l_model.prms_def.keys()])
            l_sensors = simplified_list_string(
                l_model.forward_model.output_sensor_names
            )
            exp_name_1 = l_model.experiment_name
            rows_like.append((l_name, prms_glob, l_sensors, exp_name_1))
        headers = ["Name", "Parameters", "Target sensors", "Experiment"]
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
        an instance of InverseProblem. See self.info for more details.
        """
        return self.info(return_string=True)  # type: ignore

    def check_problem_consistency(self):
        """
        Conducts various checks to make sure the problem definition does not contain
        any inconsistencies.
        """

        # check if the central components have been added to the problem: parameters,
        # forward models, experiments and likelihood models
        assert len(self.parameters) != 0, "The problem does not contain any parameters!"
        assert (
            len(self.forward_models) != 0
        ), "The problem does not contain any forward models!"
        assert (
            len(self.experiments) != 0
        ), "The problem does not contain any experiments!"
        assert (
            len(self.likelihood_models) != 0
        ), "The problem does not contain any likelihood models!"

        # check if all parameters of the forward model(s) appear in self.parameters
        # and if they have the correct type
        for fwd_name, forward_model in self.forward_models.items():
            for model_prm in forward_model.prms_def:
                assert model_prm in self.parameters, (
                    f"The forward model parameter '{model_prm}' that appears in "
                    f"forward model '{fwd_name}' is not defined within the problem!"
                )
                assert self.parameters[model_prm].type == "model", (
                    f"The forward model parameter '{model_prm}' that appears in "
                    f"forward model '{fwd_name}' has not been defined as of type "
                    f"'model' but of type '{self.parameters[model_prm].type}' in the "
                    f"problem definition!"
                )

        # check if all parameters of the likelihood model appear in self.parameters
        # and if they have the correct type
        for likelihood_name, likelihood_model in self.likelihood_models.items():
            for likelihood_prm in likelihood_model.prms_def:
                assert likelihood_prm in self.parameters, (
                    f"The likelihood model parameter '{likelihood_prm}' that appears "
                    f"in likelihood model '{likelihood_name}' is not defined within "
                    f"the problem!"
                )
                assert self.parameters[likelihood_prm].type == "likelihood", (
                    f"The likelihood model parameter '{likelihood_prm}' that appears "
                    f"in likelihood model '{likelihood_name}' has not been defined "
                    f"as of type 'likelihood' but of type "
                    f"'{self.parameters[likelihood_prm].type}' in the problem "
                    f"definition!"
                )

        # check if all prior objects in self.priors are consistent in terms of their
        # parameters; each one of them must appear in self.parameters
        for prior_name, prior_obj in self.priors.items():
            for prior_prm in prior_obj.hyperparameters:
                assert prior_prm in self.parameters, (
                    f"The prior parameter '{prior_prm}' that appears in prior "
                    f"'{prior_name}' is not defined within the problem!"
                )
                assert self.parameters[prior_prm].type == "prior", (
                    f"The prior parameter '{prior_prm}' that appears in prior "
                    f"'{prior_name}' has not been defined as of type 'prior' but of "
                    f"type '{self.parameters[prior_prm].type}' in the problem "
                    f"definition!"
                )

        # check the indices of the latent parameters; the combined indices should form
        # an ascending vector of integers from 0 to n; no multiple or missing values
        # are allowed
        idx_list = []
        for prm_name, parameter in self.parameters.items():
            if parameter.is_latent:
                idx_range = [*range(parameter.index, parameter.index_end)]
                idx_list += idx_range
        assert sorted(idx_list) == list(range(len(idx_list))), (
            f"There seems to be an inconsistency in the latent parameter's indices. "
            f"The sorted index list is: {sorted(idx_list)}"
        )

        # check the consistency of each parameter
        for parameter in self.parameters.values():
            parameter.check_consistency()

        # check that each defined experiment appears in one of the likelihood models
        exp_names_in_likelihood_models = set()
        for likelihood_model in self.likelihood_models.values():
            exp_names_in_likelihood_models.add(likelihood_model.experiment_name)
        for exp_name in self.experiments.keys():
            if exp_name not in exp_names_in_likelihood_models:
                logger.warning(
                    f"The globally defined experiment '{exp_name}' does not appear in "
                    f"any of the likelihood models!"
                )
