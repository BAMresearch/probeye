# standard library
from typing import Union, List, Optional, Callable
import copy as cp

# third party imports
from tabulate import tabulate
from loguru import logger
import numpy as np
import torch as th

# local imports
from probeye.definition.parameter import Parameters
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.noise_model import NoiseModelBase
from probeye.subroutines import underlined_string, titled_table
from probeye.subroutines import simplified_list_string, simplified_dict_string
from probeye.subroutines import unvectorize_dict_values, make_list, len_or_one
from probeye.subroutines import print_probeye_header
from probeye.subroutines import logging_setup
from probeye.subroutines import add_index_to_tex_prm_name


class InferenceProblem:
    """
    This class provides a general framework for defining an inference problem.
    Capabilities for solving the set up problem are intentionally not included.
    """

    def __init__(
        self, name: str, use_default_logger: bool = True, log_file: Optional[str] = None
    ):
        """
        Parameters
        ----------
        name
            This is the name of the problem and has only descriptive value, for example
            when working with several inference problems.
        use_default_logger
            When True, the logger will be set up with some useful default values.
            Otherwise, no logger configurations are applied.
        log_file
            Path to the log-file, if the logging should be printed to file. If None is
            given, no logging-file will be created.
        """

        # the name of the problem
        self.name = name

        # this is the central parameter dictionary of the problem (the used Parameters-
        # class is derived from the dict-class); it contains all defined parameters
        # ('const' and 'latent' ones); the keys of this dictionary are the parameter
        # names; note that each parameter must have a unique name in the problem; the
        # values of this dictionary are Parameter-objects (see parameter.py) with the
        # following attributes:
        # .index  int or None (the index in the theta-vector (see self.loglike) for
        #         'latent'-parameter; None for 'const'-parameters)
        # .type   string (either 'model', 'prior' or 'noise' depending on where the
        #         parameter appears)
        # .prior  object or None (the prior-object of the 'latent'-parameter; None for
        #         'const'-parameters)
        # .value  float or None (defines the value for 'const'-parameters; None for
        #         'latent'-parameters)
        # .info   string (a short explanation of the parameter)
        # .tex:   string or None (the TeX version of the parameter's name, for example
        #         r'$\alpha$' for a parameter named 'alpha')
        self._parameters = Parameters()

        # this dictionary is intended for storing the measured data from experiment_
        # names (see self.add_experiment); this dict is managed internally and should
        # not be edited directly
        self._experiments = {}  # type: dict

        # here, the forward models are written to; note that the problem can have
        # multiple forward models; the keys are the forward model names, while the
        # values are the forward model objects, see also in the script forward_model.py;
        # this dictionary is managed internally and should not be edited directly
        self._forward_models = {}  # type: dict

        # a list for the problem's noise models (it's not a dict like the other
        # attributes from above as the noise models don't need to have names); this
        # list is managed internally and should not be edited directly
        self._noise_models = list()  # type: List[NoiseModelBase]

        # setup the logger with the given specifications
        if use_default_logger:
            logging_setup(log_file=log_file)

        # log probeye header and first message
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
    def n_noise_prms(self) -> int:
        """Provides n_noise_prms attribute."""
        return self._parameters.n_noise_prms

    @property
    def noise_prms(self) -> List[str]:
        """Provides noise_prms attribute."""
        return self._parameters.noise_prms

    @property
    def parameters(self) -> Parameters:
        """Access self._parameters from outside via self.parameters."""
        return self._parameters

    @property
    def priors(self) -> dict:
        """Provides the problem's prior-dictionary which is derived from the
        latent parameters in the self.parameters dictionary. The keys are the
        priors names, while the values are the prior-objects."""
        return {
            parameter_properties.prior.name: parameter_properties.prior
            for parameter_properties in self.parameters.values()
            if parameter_properties.is_latent
        }

    @property
    def noise_models(self) -> list:
        """Access self._noise_models from outside via self.noise_models."""
        return self._noise_models

    @property
    def forward_models(self) -> dict:
        """Access self._forward_models from outside via self.forward_models."""
        return self._forward_models

    @property
    def experiments(self) -> dict:
        """Access self._experiments from outside via self.experiments."""
        return self._experiments

    def info(
        self,
        include_experiments: bool = False,
        tablefmt: str = "presto",
        check_consistency: bool = True,
    ) -> str:
        """
        Either prints the problem definition to the console (print_it=True) or just
        returns the generated string without printing it (print_it=False).

        Parameters
        ----------
        include_experiments
            If True, information on the experiments defined within the model will be
            included in the printout. Depending on the number of defined experiments,
            this might result in a long additional printout, which is why this is set
            to False (no experiment printout) by default.
        tablefmt
            An argument for the tabulate function defining the style of the generated
            table. Check out tabulate's documentation for more info.
        check_consistency
            When True, a consistency check is performed before printing the explanations
            on theta. When False, this check is skipped.

        Returns
        -------
            The constructed string providing the problem information.
        """

        # contains the name of the inference problem
        title = f"Problem summary: {self.name}"
        title_string = underlined_string(title, n_empty_start=1)

        # list the forward models that have been defined within the problem
        rows = [
            (
                name,
                simplified_list_string([*model.prms_def.keys()]),
                simplified_list_string([*model.prms_def.values()]),
            )
            for name, model in self._forward_models.items()
        ]
        headers = ["Model name", "Global parameters", "Local parameters"]
        fwd_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        fwd_string = titled_table("Forward models", fwd_table)

        # include information on the defined priors
        rows = [
            (
                name,
                simplified_list_string([*prior.prms_def.keys()]),
                simplified_list_string([*prior.prms_def.values()]),
            )
            for name, prior in self.priors.items()
        ]
        headers = ["Prior name", "Global parameters", "Local parameters"]
        prior_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prior_str = titled_table("Priors", prior_table)

        # provide various information on the problem's parameters
        prm_string = self._parameters.parameter_overview(tablefmt=tablefmt)
        prm_string += self._parameters.parameter_explanations(tablefmt=tablefmt)
        prm_string += self._parameters.const_parameter_values(tablefmt=tablefmt)

        # include the information on the theta interpretation
        theta_string = "\nTheta interpretation"
        theta_string += self.theta_explanation(check_consistency=check_consistency)

        # print information on added experiments if requested
        if include_experiments:
            rows_exp = []  # type: List[tuple[str, str]]
            for name, exp_dict in self._experiments.items():
                dict_atoms = unvectorize_dict_values(exp_dict["sensor_values"])
                for dict_atom in dict_atoms:
                    rows_exp.append((name, simplified_dict_string(dict_atom)))
            headers = ["Name", "Sensor values"]
            exp_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
            exp_str = titled_table("Added experiments", exp_table)
        else:
            exp_str = ""

        # concatenate the string and return it
        full_string = title_string + fwd_string + prior_str + prm_string
        full_string += theta_string + exp_str

        # log and return the string
        for line in full_string.split("\n"):
            logger.info(line)
        return full_string

    def __str__(self) -> str:
        """
        Allows to print the problem definition via print(problem) if problem is
        an instance of InferenceProblem. See self.info for more details.
        """
        return self.info()

    def add_parameter(
        self,
        prm_name: str,
        prm_type: str,
        dim: Optional[int] = 1,
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
        # are conducted in the Parameters.add_parameter method
        self._parameters.add_parameter(
            prm_name, prm_type, dim=dim, const=const, prior=prior, info=info, tex=tex
        )

    def remove_parameter(self, prm_name: str):
        """
        Removes a parameter ('const' or 'latent') from inference problem.

        Parameters
        ----------
        prm_name
            The name of the parameter to be removed.
        """

        # checks/additional actions are done by Parameters' __delitem__ method
        del self._parameters[prm_name]

    def change_parameter_role(
        self,
        prm_name: str,
        const: Union[int, float, None] = None,
        prior: Union[tuple, None] = None,
    ):
        """
        Performs the necessary tasks to change a parameter's role in the problem
        definition. A parameter's role can either be changed from 'const' to 'latent'
        or from 'latent' to 'const'.

        Parameters
        ----------
        prm_name
            The name of the parameter whose role should be changed.
        const
            If the new role is 'const', the corresponding value has to be specified by
            this argument.
        prior
            If the added parameter is a 'latent'-parameter, this argument has to be
            given as a 2-tuple. The first element (a string) defines the prior-type
            (will be referenced in inference routines). The 2nd element must be a
            dictionary stating the prior's parameters as keys and their numeric values
            as values or the name of a pre-defined parameter within the problem scope.
            An example for a normal prior: ('normal', {'loc': 0.0, 'scale': 1.0}). In
            order to define the prior's parameters, check out the prior definitions
            in priors.py.
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
        # ('model', 'prior' or 'noise')
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
            const=const,
            prior=prior,
            info=prm.info,
            tex=prm.tex,
        )

    def change_parameter_info(
        self,
        prm_name: str,
        new_info: Optional[str] = None,
        new_tex: Optional[str] = None,
    ):
        """
        Changes the info-string and/or the tex-string of a given parameter.

        Parameters
        ----------
        prm_name
            The name of the parameter whose info-string should be changed.
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
        inference problem.

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

    def check_problem_consistency(self):
        """
        Conducts various checks to make sure the problem definition does not contain
        any inconsistencies.
        """

        # check if the central components have been added to the problem: parameters,
        # forward models, noise models and experiments
        assert (
            len(self._parameters) != 0
        ), "The problem does not contain any parameters!"
        assert (
            len(self._forward_models) != 0
        ), "The problem does not contain any forward models!"
        assert (
            len(self._noise_models) != 0
        ), "The problem does not contain any noise models!"
        assert (
            len(self._experiments) != 0
        ), "The problem does not contain any experiments!"

        # check if all parameters of the forward model(s) appear in self._parameters
        # and if they have the correct type
        for forward_model in self._forward_models.values():
            for model_prm in forward_model.prms_def.keys():
                assert model_prm in self._parameters, (
                    f"The forward model parameter '{model_prm}' is not defined "
                    f"within the problem!"
                )
                assert (
                    self._parameters[model_prm].type == "model"
                ), f"The forward model parameter '{model_prm}' is not of type 'model'!"

        # check if all parameters of the noise model appear in self._parameters
        # and if they have the correct type
        for noise_model in self._noise_models:
            for noise_prm in noise_model.prms_def.keys():
                assert noise_prm in self._parameters, (
                    f"The noise model parameter '{noise_prm}' is not defined within "
                    f"the problem!"
                )
                assert (
                    self._parameters[noise_prm].type == "noise"
                ), f"The noise model parameter '{noise_prm}' is not of type 'noise'!"

        # check if all prior objects in self.priors are consistent in terms of their
        # parameters; each one of them must appear in self._parameters
        for prior_obj in self.priors.values():
            for prior_prm in prior_obj.hyperparameters.keys():
                assert prior_prm in self._parameters, (
                    f"The prior parameter '{prior_prm}' is not defined within "
                    f"the problem!"
                )
                assert self._parameters[prior_prm].type == "prior", (
                    f"The prior parameter '{prior_prm}' " f"is not of type 'prior'!"
                )

        # check the indices of the latent parameters
        idx_list = []
        for prm_name, parameter in self._parameters.items():
            if parameter.is_latent:
                idx_range = [*range(parameter.index, parameter.index_end)]
                idx_list += idx_range
        assert sorted(idx_list) == list(range(len(idx_list))), (
            f"There seems to be an inconsistency in the latent parameter's indices. "
            f"The sorted index list is: "
            f"{sorted(idx_list)}"
        )

        # check the consistency of each parameter
        for parameter in self._parameters.values():
            parameter.check_consistency()

    def add_experiment(self, exp_name: str, sensor_values: dict, fwd_model_name: str):
        """
        Adds a single experiment to the inference problem. Here, an experiment is
        defined as one or more sensors (note that the experiment does not distinguish
        between input and output sensors) which provide some measured data.
        Additionally, a reference to one of the problem's forward models is associated
        with an experiment.

        Parameters
        ----------
        exp_name
            The name of the experiment, e.g. "Exp_20May.12". If an experiment with a
            similar name has already been added, it will be overwritten and a warning
            will be thrown.
        sensor_values
            The keys are the sensor's names, the values are the measured values.
        fwd_model_name
            Name of the forward model this experiment refers to.
        """

        # log at beginning so that errors can be associated
        logger.debug(f"Adding experiment '{exp_name}'")

        # check all keyword arguments are given
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

        # check if the given forward model exists
        if fwd_model_name not in self._forward_models.keys():
            raise RuntimeError(
                f"The forward model '{fwd_model_name}' does not exist! You need to "
                f"define it before adding experiments that refer to it."
            )

        # check that the stated forward model is consistent with the experiment
        experiment_sensors = [*sensor_values.keys()]
        input_sensors = self._forward_models[fwd_model_name].input_sensors
        for input_sensor in input_sensors:
            if input_sensor.name not in experiment_sensors:
                raise RuntimeError(
                    f"The forward model's ({fwd_model_name}) input sensor "
                    f"'{input_sensor.name}' is not provided by the given "
                    f"experiment '{exp_name}'!"
                )
        output_sensors = self._forward_models[fwd_model_name].output_sensors
        for output_sensor in output_sensors:
            if output_sensor.name not in experiment_sensors:
                raise RuntimeError(
                    f"The forward model's ({fwd_model_name}) output sensor "
                    f"'{output_sensor.name}' is not provided by the given "
                    f"experiment '{exp_name}'!"
                )

        # check that vector-valued sensor_values are given as numpy-arrays; if
        # not (e.g. if lists or tuples are given) change them to numpy-ndarrays
        sensor_values_numpy = cp.copy(sensor_values)
        for sensor_name, values in sensor_values.items():
            if hasattr(values, "__len__"):
                if not isinstance(values, np.ndarray):
                    sensor_values_numpy[sensor_name] = np.array(values)

        # throw warning when the experiment name was defined before
        if exp_name in self._experiments.keys():
            logger.warning(
                f"Experiment '{exp_name}' is already defined "
                f"and will be overwritten!"
            )

        # add the experiment to the central dictionary
        self._experiments[exp_name] = {
            "sensor_values": sensor_values_numpy,
            "forward_model": fwd_model_name,
        }

    def get_parameters(
        self, theta: Union[np.ndarray, th.Tensor], prm_def: dict
    ) -> dict:
        """
        Extracts the numeric values for given parameters from the parameter vector theta
        and the constant parameters of the problem.

        Parameters
        ----------
        theta
            A numeric parameter vector passed to the loglike and logprior method. Which
            parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        prm_def
            Defines which parameters to extract. The keys of this dictionary are the
            global parameter names, while the values are the local parameter names. In
            most cases global and local names will be identical, but sometimes it is
            convenient to define a local parameter name, e.g. in the forward model.

        Returns
        -------
        prms
            Contains <local parameter name> : <(global) parameter value> pairs.
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

    def get_experiment_names(
        self,
        forward_model_names: Union[str, List[str], None] = None,
        sensor_names: Union[str, List[str], None] = None,
        experiment_names: Union[str, List[str], None] = None,
    ) -> list:
        """
        Extracts the names of all experiments which refer to a given list of forward
        models and/or to a given list of sensor names from a given list of
        experiment names.

        Parameters
        ----------
        forward_model_names
            The names of the forward model the experiments should refer to.
        sensor_names
            The names of the sensors the experiments should should contain.
        experiment_names
            The names of the experiments to sub-select from. If None is given, then all
            experiments of the problem will be used.

        Returns
        -------
        relevant_experiment_names
            The names of the sub-selected experiments.
        """

        # at least one of forward_model_names and sensor_names must be given
        if (forward_model_names is None) and (sensor_names is None):
            raise RuntimeError(
                f"You did not specify any forward model(s) or sensor name(s)."
            )

        # if experiments is not further specified it is assumed that all given
        # experiments should be used
        if experiment_names is None:
            experiment_names = [*self._experiments.keys()]

        # this is for collecting the experiments
        relevant_experiment_names = []

        # get the experiments which refer to the given forward models
        if forward_model_names is not None:
            forward_model_names_list = make_list(forward_model_names)
            for exp_name in experiment_names:
                exp_dict = self._experiments[exp_name]
                fwd_model_name = exp_dict["forward_model"]
                if fwd_model_name in forward_model_names_list:
                    relevant_experiment_names.append(exp_name)
            experiment_names = relevant_experiment_names

        # get the experiments which contain the given sensors
        if sensor_names is not None:
            relevant_experiment_names = []
            sensor_names_list = make_list(sensor_names)
            for exp_name in experiment_names:
                exp_dict = self._experiments[exp_name]
                exp_sensors = [*exp_dict["sensor_values"].keys()]
                if all([s in exp_sensors for s in sensor_names_list]):
                    relevant_experiment_names.append(exp_name)

        return relevant_experiment_names

    def get_theta_names(self, tex: bool = False, components: bool = False) -> list:
        """
        Returns the parameter names of the parameter vector theta in the corresponding
        order (either in tex- or conventional format).

        Parameters
        ----------
        tex
            If True, the TeX-names of the parameters will be returned, otherwise the
            names as they are used in the code will be returned.
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
        Logs and returns a string on how the theta-vector, which is the numeric
        parameter vector that is given to the self.loglike and self.logprior methods,
        is interpreted with respect to the problem's parameters. The printout will tell
        you which parameter is connected to which index of theta.

        Parameters
        ----------
        check_consistency
            When True, a consistency check is performed before printing the
            explanations on theta. When False, this check is skipped.

        Returns
        -------
        s
            The constructed string when 'log' was set to False.
        """

        # an explanation is not printed if the problem is inconsistent
        if check_consistency:
            self.check_problem_consistency()

        # collect the list of theta names in the right order
        theta_names = self.get_theta_names()

        # construct the info-string
        s = "\n+---------------------------+\n"
        s += "|  Theta  |    Parameter    |\n"
        s += "|  index  |      name       |\n"
        s += "|---------------------------|\n"
        for prm_name in theta_names:
            idx = self.parameters[prm_name].index
            dim = self.parameters[prm_name].dim
            if dim == 1:
                s += f"|{idx:7d} --> {prm_name:<15s}|\n"
            else:
                idx_str = f"{idx}-{idx + dim - 1}"
                s += f"|{idx_str:>7s} --> {prm_name:<15s}|\n"
        s += "+---------------------------+\n"

        return s

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
            forward model from that base class.
        """

        # log at beginning so that errors can be associated
        logger.debug(f"Adding forward model '{name}'")

        # check if all given model parameters have already been added to the inference
        # problem; note that the forward model can only be added to the problem after
        # the corresponding parameters were defined
        for prm_name in forward_model.prms_def:
            self._parameters.confirm_that_parameter_exists(prm_name)

        # check if the given name for the forward model has already been used
        if name in [*self._forward_models.keys()]:
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

    def add_noise_model(self, noise_model: NoiseModelBase):
        """
        Adds a noise model to the inference problem.

        Parameters
        ----------
        noise_model
            The noise model object, e.g. from NormalNoise. Check out noise.py to see
            some noise model classes.
        """

        # check if the noise model has been assigned a name; if not, assign one
        if noise_model.name is None:
            noise_model.name = f"noise_model_{len(self.noise_models)}"
            logger.debug(
                f"Adding noise model '{noise_model.name}' (name assigned automatically)"
            )
        else:
            logger.debug(f"Adding noise model '{noise_model.name}'")

        # check if all given noise model parameters have already been added to
        # the inference problem
        for prm_name in noise_model.prms_def:
            if prm_name not in self._parameters.keys():
                raise RuntimeError(
                    f"The noise model parameter '{prm_name}' has not been defined yet."
                    f"\nYou have to add all noise model parameters to the problem "
                    f"before adding the noise model.\nYou can use the 'add_parameter' "
                    f"method for this purpose."
                )

        # add the problem's experiments to the noise model (this is just a pointer!)
        # for noise_model-internal checks
        noise_model.problem_experiments = self._experiments
        # finally, add the noise_model to the internal dict
        self._noise_models.append(noise_model)

    def assign_experiments_to_noise_models(self):
        """
        Assigns each noise model the corresponding experiment names, based on the sensor
        names, that are defined for each noise model. This function is intended to be
        called after the problem was fully defined. Alternatively, you can assign the
        experiments 'by hand', by using the NoiseModelBase's 'add_experiments' method.
        """
        logger.debug("Assigning experiments to noise models")
        n_experiments_defined = len(self._experiments)
        n_experiments_noise = 0
        for noise_model in self._noise_models:
            if noise_model.assign_experiments_automatically:
                # get experiments that contain all of the noise model's sensors
                experiment_names = self.get_experiment_names(
                    sensor_names=noise_model.sensor_names
                )
                n_experiments_noise += len(experiment_names)
                # add the relevant experiment names to the noise model
                noise_model.add_experiments(experiment_names)
            else:
                n_experiments_noise += len(noise_model.experiment_names)
            noise_model.prepare_corr_dict()

        # check if there is the same number of experiments over all noise models as
        # defined for the inference problem
        if n_experiments_noise != n_experiments_defined:
            # this is not necessarily an error; it also happens in a valid setup when
            # more than one noise model are defined for one forward model; in a future
            # version, there could be an info message here
            pass

        # check that each globally defined experiment appears in one of the noise models
        exp_names_in_noise_models = set()
        for noise_model in self._noise_models:
            for exp_name_noise in noise_model.experiment_names:
                exp_names_in_noise_models.add(exp_name_noise)

        for exp_name in self._experiments.keys():
            if exp_name not in exp_names_in_noise_models:
                # undo the adding of the experiment names
                for noise_model in self._noise_models:
                    noise_model.experiment_names = []
                # one may argue, that this could also be only a warning here
                raise RuntimeError(
                    f"The globally defined experiment '{exp_name}' does not appear in "
                    f"any of the noise models!"
                )

    def transform_experimental_data(
        self, f: Callable, args: tuple = (), **kwargs
    ) -> "InferenceProblem":
        """
        Creates a full copy of the problem the experimental data of which is transformed
        in some way. This might be a necessary pre-processing step for an inference
        engine in order to be able to solve the problem. Note that the original problem
        remains unchanged.

        Parameters
        ----------
        f
            The function that is applied on each of the experiment's sensor values.
        args
            Additional positional arguments to be passed to f.
        kwargs
            Keyword arguments to be passed to f.

        Returns
        -------
        self_copy
            A full copy of self where the experimental data has been transformed in the
            specified fashion.
        """

        logger.debug(f"Transforming experimental data using f = '{f.__name__}'")

        # the original problem shall not be touched, so we create a copy here to which
        # the transformation will be applied; however, sometimes this is not possible,
        # hence the try-except frame
        try:
            self_copy = cp.deepcopy(self)
        except:
            logger.warning(
                "The inference problem could not be deep-copied! "
                "The original problem will be modified!"
            )
            self_copy = self

        # transform the sensor values from the experiments by applying the specified
        # function with the given arguments to them
        for exp_name in self_copy._experiments.keys():
            sensor_values = self_copy._experiments[exp_name]["sensor_values"]
            for sensor_name in sensor_values.keys():
                sensor_values[sensor_name] = f(
                    sensor_values[sensor_name], *args, **kwargs
                )

        return self_copy
