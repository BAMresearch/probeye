# standard library
from copy import copy

# third party imports
from tabulate import tabulate
import numpy as np

# local imports
from probeye.definition.parameter import Parameters, ParameterProperties
from probeye.definition.prior import PriorTemplate
from probeye.subroutines import underlined_string, titled_table
from probeye.subroutines import simplified_list_string, simplified_dict_string
from probeye.subroutines import unvectorize_dict_values, make_list, len_or_one


class InferenceProblem:
    """
    This class provides a general framework for defining an inference problem.
    Capabilities for solving the set up problem are intentionally not included.
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            This is the name of the problem and has only descriptive value, for
            example when working with several inference problems.
        """

        # the name of the problem
        self.name = name

        # this is the central parameter dictionary of the problem (the used
        # Parameters-class is derived from the dict-class); it contains all
        # defined parameters ('const' and 'calibration' ones); the keys of this
        # dictionary are the parameter names; note that each parameter must have
        # a unique name in the problem; the values of this dictionary are
        # Parameter-objects (see parameter.py) with the following attributes:
        # .index  int or None (the index in the theta-vector (see self.loglike)
        #         for 'calibration'-parameter; None for 'const'-parameters)
        # .type   string (either 'model', 'prior' or 'noise' depending on where
        #         the parameter appears)
        # .role   string (either 'const' for a constant parameter or
        #         'calibration' for a calibration parameter)
        # .prior  object or None (the prior-object of the 'calibration'-
        #         parameter; None for 'const'-parameters)
        # .value  float or None (defines the value for 'const'-parameters;
        #         None for 'calibration'-parameters)
        # .info   string (a short explanation of the parameter)
        # .tex:   string or None (the TeX version of the parameter's name, for
        #         example r'$\alpha$' for a parameter named 'alpha')
        self._parameters = Parameters()

        # this dictionary is intended for storing the measured data from
        # experiment_names (see self.add_experiment); this dict is managed
        # internally and should not be edited directly
        self._experiments = {}

        # dictionary of the problem's priors; the items will have the structure
        # <prior name> : <prior object>; this dict is managed internally and
        # should not be edited directly
        self._priors = {}

        # here, the forward models are written to; note that the problem can
        # have multiple forward models; the keys are the forward model names,
        # while the values are the forward model objects, see also in the script
        # forward_model.py; this dictionary is managed internally and should not
        # be edited directly
        self._forward_models = {}

        # a dictionary for the problem's noise models; note that noise models
        # are defined sensor-specific, so the items of this dict are of the
        # structure <sensor name> : <noise model object>; this dict is managed
        # internally and should not be edited directly
        self._noise_models = {}

    @property
    def n_prms(self):
        """Provides n_prms attribute."""
        return self._parameters.n_prms

    @property
    def prms(self):
        """Provides prms attribute."""
        return self._parameters.prms

    @property
    def n_calibration_prms(self):
        """Provides n_calibration_prms attribute."""
        return self._parameters.n_calibration_prms

    @property
    def calibration_prms(self):
        """Provides calibration_prms attribute."""
        return self._parameters.calibration_prms

    @property
    def n_constant_prms(self):
        """Provides n_constant_prms attribute."""
        return self._parameters.n_constant_prms

    @property
    def constant_prms(self):
        """Provides constant_prms attribute."""
        return self._parameters.constant_prms

    @property
    def n_model_prms(self):
        """Provides n_model_prms attribute."""
        return self._parameters.n_model_prms

    @property
    def model_prms(self):
        """Provides model_prms attribute."""
        return self._parameters.model_prms

    @property
    def n_prior_prms(self):
        """Provides n_prior_prms attribute."""
        return self._parameters.n_prior_prms

    @property
    def prior_prms(self):
        """Provides prior_prms attribute."""
        return self._parameters.prior_prms

    @property
    def n_noise_prms(self):
        """Provides n_noise_prms attribute."""
        return self._parameters.n_noise_prms

    @property
    def noise_prms(self):
        """Provides noise_prms attribute."""
        return self._parameters.noise_prms

    @property
    def parameters(self):
        """Access self._parameters from outside via self.parameters."""
        return self._parameters

    @property
    def priors(self):
        """Access self._priors from outside via self.priors."""
        return self._priors

    @property
    def noise_models(self):
        """Access self._noise_models from outside via self.noise_models."""
        return self._noise_models

    @property
    def experiments(self):
        """Access self._experiments from outside via self.experiments."""
        return self._experiments

    def info(self, print_it=True, include_experiments=False, tablefmt="presto",
             check_consistency=True):
        """
        Either prints the problem definition to the console (print_it=True) or
        just returns the generated string without printing it (print_it=False).

        Parameters
        ----------
        print_it : bool, optional
            If True, the generated string is printed and not returned. If set
            to False, the generated string is not printed but returned.
        include_experiments : bool, optional
            If True, information on the experiments defined within the model
            will be included in the printout. Depending on the number of defined
            experiments, this might result in a long additional printout, which
            is why this is set to False (no experiment printout) by default.
        tablefmt : str, optional
            An argument for the tabulate function defining the style of the
            generated table. Check out tabulate's documentation for more info.
        check_consistency : bool, optional
            When True, a consistency check is performed before printing the
            explanations on theta. When False, this check is skipped.

        Returns
        -------
        string or None
            The constructed string when 'print_it' was set to False.
        """

        # contains the name of the inference problem
        title_string = underlined_string(self.name, n_empty_start=2)

        # list the forward models that have been defined within the problem
        rows = [(name, simplified_list_string([*model.prms_def.keys()]),
                simplified_list_string([*model.prms_def.values()]))
                for name, model in self._forward_models.items()]
        headers = ["Model name", "Global parameters", "Local parameters"]
        fwd_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        fwd_string = titled_table('Forward models', fwd_table)

        # include information on the defined priors
        rows = [(name, simplified_list_string([*prior.prms_def.keys()]),
                 simplified_list_string([*prior.prms_def.values()]))
                for name, prior in self._priors.items()]
        headers = ["Prior name", "Global parameters", "Local parameters"]
        prior_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        prior_str = titled_table('Priors', prior_table)

        # provide various information on the problem's parameters
        prm_string = self._parameters.parameter_overview(tablefmt=tablefmt)
        prm_string += self._parameters.parameter_explanations(tablefmt=tablefmt)
        prm_string += self._parameters.const_parameter_values(tablefmt=tablefmt)

        # include the information on the theta interpretation
        theta_string = "\nTheta interpretation"
        theta_string += self.theta_explanation(
            print_it=False, check_consistency=check_consistency)

        # print information on added experiments if requested
        if include_experiments:
            rows = []
            for name, exp_dict in self._experiments.items():
                dict_atoms = unvectorize_dict_values(exp_dict['sensor_values'])
                for dict_atom in dict_atoms:
                    rows.append((name, simplified_dict_string(dict_atom)))
            headers = ["Name", "Sensor values"]
            exp_table = tabulate(rows, headers=headers, tablefmt=tablefmt)
            exp_str = titled_table('Added experiments', exp_table)
        else:
            exp_str = ""

        # concatenate the string and return it
        full_string = title_string + fwd_string + prior_str + prm_string
        full_string += theta_string + exp_str

        # either print or return the string
        if print_it:
            print(full_string)
        else:
            return full_string

    def __str__(self):
        """
        Allows to print the problem definition via print(problem) if problem is
        an instance of InferenceProblem. See self.info for more details.
        """
        return self.info(print_it=False)

    def add_parameter(self, prm_name, prm_type, const=None, prior=None,
                      info="No explanation provided", tex=None):
        """
        Adds a parameter ('const' or 'calibration') to the inference problem.

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
            If the added parameter is a 'calibration'-parameter, this argument
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

        # make sure the given prm_type is valid
        if prm_type not in ['model', 'prior', 'noise']:
            raise RuntimeError(
                f"Unknown parameter type: prm_type = {prm_type} \n" +
                f"Valid arguments are 'model', 'prior' or 'noise'."
            )

        # exactly one of the const and prior key word arguments must be given
        if const is not None and prior is not None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified both."
            )
        if const is None and prior is None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified none."
            )

        # check whether the parameter name was used before; note that all
        # parameters (across types!) must have unique names
        if prm_name in self._parameters.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has already been" +
                f" defined. Please choose another name."
            )

        # add the parameter to the central parameter dictionary
        prm_role = 'calibration' if const is None else 'const'
        if const is None:  # in this case we are adding a 'calibration'-param.
            # first, define the index of this parameter in the numeric vector
            # theta, which is given to self.loglike and self.logprior
            prm_index = self._parameters.n_calibration_prms
            # the prm_value is reserved for 'const'-parameter; hence, it is set
            # to None in this case, where we are adding a 'calibration'-param.
            prm_value = None
            # the remaining code in this if-branch defines the prior that is
            # associated with this 'calibration'-parameter
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
                    default_info += f"for calibration-parameter '{prm_name}'"
                    self.add_parameter(new_name, 'prior', const=value,
                                       info=default_info)  # recursive call
                elif type(value) is str:
                    # in this case the prior-parameter is defined as an already
                    # defined parameter with the name stated in value
                    self.check_if_parameter_exists(value)
                else:
                    raise TypeError(
                        f"The prior-parameter {new_name} is not assigned a "
                        f"float, int or str, but something of type "
                        f"{type(value)}."
                    )
            prior_name = f"{prm_name}_{prior_type}"  # unique name of this prior
            prm_prior = self._add_prior(prior_name, prior_type,
                                        prior_parameter_names, prm_name)
        else:
            # in this case we are adding a 'const'-parameter, which means that
            # the prm_index and prm_prior values are not used here
            prm_index = None
            prm_prior = None
            prm_value = const

        # add the parameter to the central parameter dictionary
        self._parameters[prm_name] = ParameterProperties({'index': prm_index,
                                                          'type': prm_type,
                                                          'role': prm_role,
                                                          'prior': prm_prior,
                                                          'value': prm_value,
                                                          'info': info,
                                                          'tex': tex})

    def remove_parameter(self, prm_name):
        """
        Removes a parameter ('const' or 'calibration') from inference problem.

        Parameters
        ----------
        prm_name : str
            The name of the parameter to be removed.
        """
        # check if the given parameter exists
        self.check_if_parameter_exists(prm_name)

        # different steps must be taken depending on whether the parameter which
        # should be removed is a 'const'- or a 'calibration'-parameter
        if self._parameters[prm_name].index is None:
            # in this case prm_name refers to a constant parameter; hence, we
            # can simply remove this parameter without side effects
            del self._parameters[prm_name]
        else:
            # in this case prm_name refers to a calibration parameter; hence we
            # need to remove the prior-parameter and the prior-object; also, we
            # have to correct the index values of the remaining calibration prms
            for prior_prm in self._parameters[prm_name].prior.\
                    prms_def_no_ref.keys():
                self.remove_parameter(prior_prm)  # recursive call
            del self._priors[self._parameters[prm_name].prior.name]
            del self._parameters[prm_name]
            # correct the indices of the remaining 'calibration'-parameters
            idx = 0
            for name, parameter in self._parameters.items():
                if parameter.index is not None:
                    parameter.index = idx
                    idx += 1

    def check_if_parameter_exists(self, prm_name):
        """
        Checks if a parameter, given by its name, exists within the problem. An
        error is raised when the given parameter does not exist yet.

        Parameters
        ----------
        prm_name : str
            A global parameter name.
        """
        # check if the given parameter exists
        if prm_name not in self._parameters.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet.")

    def change_parameter_role(self, prm_name, const=None, prior=None,
                              new_info=None, new_tex=None):
        """
        Performs the necessary tasks to change a parameter's role in the problem
        definition. A parameter's role can either be changed from 'const' to
        'calibration' or from 'calibration' to 'const'.

        Parameters
        ----------
        prm_name : str
            The name of the parameter whose role should be changed.
        const : float or None, optional
            If the new role is 'const', the corresponding value has to be
            specified by this argument.
        prior : tuple of two elements or None, optional
            If the added parameter is a 'calibration'-parameter, this argument
            has to be given as a 2-tuple. The first element (a string) defines
            the prior-type (will be referenced in inference routines). The 2nd
            element must be a dictionary stating the prior's parameters as keys
            and their numeric values as values or the name of a pre-defined
            parameter within the problem scope. An example for a normal prior:
            ('normal', {'loc': 0.0, 'scale': 1.0}). In order to define the
            prior's parameters, check out the prior definitions in priors.py.
        new_info : str or None, optional
            The new string for the explanation of parameter prm_name.
        new_tex : str or None, optional
            The new string for the parameter's tex-representation.
        """
        # check if the given parameter exists
        self.check_if_parameter_exists(prm_name)

        # exactly one of the const and prior key word arguments must be given
        if const is not None and prior is not None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified both."
            )
        if const is None and prior is None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified none."
            )
        # raise an error if the role change would not change the role
        current_role = self._parameters[prm_name].role
        if (current_role == 'const') and (prior is None):
            raise RuntimeError(
                f"The parameter '{prm_name}' is already defined as constant."
            )
        if (current_role == 'calibration') and (const is None):
            raise RuntimeError(
                f"The parameter '{prm_name}' is already defined as a "
                f"calibration parameter."
            )
        # the parameter's role is changed by first removing it from the problem,
        # and then adding it again in its new role; the role-change does not
        # impact the type ('model', 'prior' or 'noise')
        prm_type = self._parameters[prm_name].type
        # if no new_info/new_tex was specified, use the old ones
        if new_info is None:
            prm_info = self._parameters[prm_name].info
        else:
            prm_info = new_info
        if new_tex is None:
            prm_tex = self._parameters[prm_name].tex
        else:
            prm_tex = new_tex
        # now we can finally change the role
        self.remove_parameter(prm_name)
        self.add_parameter(prm_name, prm_type, const=const, prior=prior,
                           info=prm_info, tex=prm_tex)

    def change_parameter_info(self, prm_name, new_info, new_tex=None):
        """
        Changes the info-string and/or the tex-string of a given parameter.

        Parameters
        ----------
        prm_name : str
            The name of the parameter whose info-string should be changed.
        new_info : str
            The new string for the explanation of parameter prm_name.
        new_tex : str or None
            The new string for the parameter's tex-representation.
        """
        # check if the given parameter exists
        self.check_if_parameter_exists(prm_name)

        # change the info/tex-string
        self._parameters[prm_name].info = new_info
        if new_tex is not None:
            self._parameters[prm_name].tex = new_tex

    def change_constant(self, prm_name, new_value):
        """
        Changes the value of a 'const'-parameter, i.e. a constant parameter of
        the inference problem.

        Parameters
        ----------
        prm_name : str
            The name of the 'const'-parameter whose value should be changed.
        new_value : float
            The new value that prm_name should assume.
        """
        # check if the given parameter exists
        self.check_if_parameter_exists(prm_name)

        # check if the given parameter is a constant
        if self._parameters[prm_name].role != "const":
            raise RuntimeError(
                f"The parameter '{prm_name}' is not a constant!"
            )
        # change the parameter's value
        self._parameters[prm_name].value = new_value

    def _add_prior(self, name, prior_type, prms_def, ref_prm):
        """
        Adds a PriorTemplate-object, generally representing a prior of a
        calibration parameter to the internal prior dictionary. In the inference
        step, after defining the problem, this template object has to be
        translated into a prior-object of the user's choice, that is able to
        evaluate functions like the logpdf. The PriorTemplate does not have this
        capabilities, it merely describes the prior-type, its parameters, etc.

        Parameters
        ----------
        name : str
            Unique name of the prior. Usually this name has the structure
            <ref_prm>_<prior_type>.
        prior_type : str
            Defines the prior type, e.g. 'normal' or 'uniform'.
        prms_def : list[str]
            States the prior's parameter names.
        ref_prm : str
            The name of the problem's calibration parameter the prior refers to
            (a prior is always defined for a specific calibration parameter).

        Returns
        -------
        obj[PriorTemplate]
            The instantiated PriorTemplate-object which is also written to the
            internal prior dictionary self._priors.
        """
        # check if the prior parameters exist (a prior cannot be defined before
        # its parameters have been defined)
        for prior_parameter in prms_def:
            self.check_if_parameter_exists(prior_parameter)

        # check if a prior with the same name was defined before
        if name in [*self._priors.keys()]:
            raise RuntimeError(
                f"A prior with the name '{name}' already exists!")

        # add the prior to the internal dictionary
        self._priors[name] = PriorTemplate(ref_prm, prms_def, name, prior_type)
        return self._priors[name]

    def check_problem_consistency(self):
        """
        Conducts various checks to make sure the problem definition does not
        contain any inconsistencies.
        """

        # check if the central components have been added to the problem:
        # parameters, priors, a forward model, experiments and noise models;
        # the following statements assert that the corresponding attributes are
        # not empty or None
        assert self._parameters, "No parameters have been defined yet!"
        assert self._priors, "Found no priors in the problem definition!"
        assert self._forward_models, "No forward model has been defined yet!"
        assert self._noise_models, "No noise models have been defined yet!"
        assert self._experiments, "No experiments have been defined yet!"

        # check if all constant parameters have values assigned
        for parameter in self._parameters.values():
            if parameter.role == 'const':
                assert parameter.value is not None

        # check if all parameters of the forward model(s) appear in
        # self._parameters and if they have the correct type
        for forward_model in self._forward_models.values():
            for model_prm in forward_model.prms_def.keys():
                assert model_prm in self._parameters.keys()
                assert self._parameters[model_prm].type == "model"

        # check if all parameters of the noise model appear in self._parameters
        # and if they have the correct type
        for noise_model in self._noise_models.values():
            for noise_prm in noise_model.prms_def.keys():
                assert noise_prm in self._parameters.keys()
                assert self._parameters[noise_prm].type == "noise"

        # check if all prior objects in self._priors are consistent in terms of
        # their parameters; each one of them must appear in self._parameters
        assert len(self._priors) == self._parameters.n_calibration_prms
        for prior_obj in self._priors.values():
            for prior_prm in prior_obj.prms_def_no_ref.keys():
                assert prior_prm in self._parameters.keys()
                assert self._parameters[prior_prm].type == 'prior'

        # check if the prior-parameters of each calibration parameter exist in
        # the problem's parameter dictionary
        for prm_name, parameter in self._parameters.items():
            if parameter.role == 'calibration':
                for prior_prm in parameter.prior.prms_def_no_ref.keys():
                    assert prior_prm in self._parameters.keys()
                    assert self._parameters[prior_prm].type == 'prior'

        # check the indices of the calibration parameters
        idx_list = []
        for prm_name, parameter in self._parameters.items():
            if parameter.role == 'calibration':
                idx_list.append(parameter.index)
        assert len(idx_list) == self._parameters.n_calibration_prms
        assert sorted(idx_list) == list(range(len(idx_list)))

    def add_experiment(self, exp_name, sensor_values, fwd_model_name):
        """
        Adds a single experiment to the inference problem. Here, an experiment
        is defined as one or more sensors (note that the experiment does not
        distinguish between input and output sensors) which provide some
        measured data. Additionally, a reference to one of the problem's forward
        models is associated with an experiment.

        Parameters
        ----------
        exp_name : str
            The name of the experiment, e.g. "Exp_20May.12". If an experiment
            with a similar name has already been added, it will be overwritten
            and a warning will be thrown.
        sensor_values : dict
            The keys are the sensor's names, the values are the measured values.
        fwd_model_name : str
            Name of the forward model this experiment refers to.
        """

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
                f"The forward model '{fwd_model_name}' does not exist! You "
                f"need to define it before adding experiments that refer to it."
            )

        # check that the stated forward model is consistent with the experiment
        experiment_sensors = [*sensor_values.keys()]
        input_sensors = self._forward_models[fwd_model_name].input_sensors
        for input_sensor in input_sensors:
            if input_sensor.name not in experiment_sensors:
                raise RuntimeError(
                    f"The forward model's ({fwd_model_name}) input sensor "
                    f"'{input_sensor.name}' is not provided by the given "
                    f"experiment '{exp_name}'!")
        output_sensors = self._forward_models[fwd_model_name].output_sensors
        for output_sensor in output_sensors:
            if output_sensor.name not in experiment_sensors:
                raise RuntimeError(
                    f"The forward model's ({fwd_model_name}) output sensor "
                    f"'{output_sensor.name}' is not provided by the given "
                    f"experiment '{exp_name}'!")

        # check if all sensor_values have the same lengths
        vector_lengths = set()
        for sensor_name, values in sensor_values.items():
            vector_lengths.add(len_or_one(values))
        if len(vector_lengths) > 1:
            raise RuntimeError(
                f"The sensor values must be all scalars or vectors of the same "
                f"length. However, found the lengths {vector_lengths}.")

        # check that vector-valued sensor_values are given as numpy-arrays; if
        # not (e.g. if lists or tuples are given) change them to numpy-ndarrays
        sensor_values_numpy = copy(sensor_values)
        for sensor_name, values in sensor_values.items():
            if hasattr(values, '__len__'):
                if not isinstance(values, np.ndarray):
                    sensor_values_numpy[sensor_name] = np.array(values)

        # throw warning when the experiment name was defined before
        if exp_name in self._experiments.keys():
            print(f"WARNING - Experiment '{exp_name}' is already defined" +
                  f" and will be overwritten!")

        # add the experiment to the central dictionary
        self._experiments[exp_name] = {'sensor_values': sensor_values_numpy,
                                       'forward_model': fwd_model_name}

    def get_parameters(self, theta, prm_def):
        """
        Extracts the numeric values for given parameters from the parameter
        vector theta and the constant parameters of the problem.

        Parameters
        ----------
        theta : array_like
            A numeric parameter vector passed to the loglike and logprior
            method. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.
        prm_def : dict
            Defines which parameters to extract. The keys of this dictionary are
            the global parameter names, while the values are the local parameter
            names. In most cases global and local names will be identical, but
            sometimes it is convenient to define a local parameter name, for
            example in the forward model.

        Returns
        -------
        prms : dict
            Contains <local parameter name> : <(global) parameter value> pairs.
        """
        prms = {}
        for global_name, local_name in prm_def.items():
            idx = self._parameters[global_name].index
            if idx is None:
                # in this case, the parameter is a constant and hence not read
                # from theta, but from the internal library
                prms[local_name] = self._parameters[global_name].value
            else:
                # in this case, the parameter is a calibration parameter, and
                # its value is read from theta
                prms[local_name] = theta[idx]
        return prms

    def get_experiment_names(self, forward_model_names=None, sensor_names=None,
                             experiment_names=None):
        """
        Extracts the names of all experiments which refer to a given list of
        forward models and/or to a given list of sensor names from a given list
        of experiment names.

        Parameters
        ----------
        forward_model_names : str, list[str] or None, optional
            The names of the forward model the experiments should refer to.
        sensor_names : list or None, optional
            The names of the sensors the experiments should should contain.
        experiment_names : str, list[str] or None, optional
            The names of the experiments to sub-select from. If None is given,
            then all experiments of the problem will be used.

        Returns
        -------
        relevant_experiment_names : list
            The names of the sub-selected experiments.
        """

        # at least one of forward_model_names and sensor_names must be given
        if (forward_model_names is None) and (sensor_names is None):
            raise RuntimeError(
                f"You did not specify any forward model(s) or sensor name(s).")

        # if experiments is not further specified it is assumed that all given
        # experiments should be used
        if experiment_names is None:
            experiment_names = [*self._experiments.keys()]

        # this is for collecting the experiments
        relevant_experiment_names = []

        # get the experiments which refer to the given forward models
        if forward_model_names is not None:
            forward_model_names = make_list(forward_model_names)
            for exp_name in experiment_names:
                exp_dict = self._experiments[exp_name]
                fwd_model_name = exp_dict['forward_model']
                if fwd_model_name in forward_model_names:
                    relevant_experiment_names.append(exp_name)
            experiment_names = relevant_experiment_names

        # get the experiments which contain the given sensors
        if sensor_names is not None:
            relevant_experiment_names = []
            sensor_names = make_list(sensor_names)
            for exp_name in experiment_names:
                exp_dict = self._experiments[exp_name]
                exp_sensors = [*exp_dict['sensor_values'].keys()]
                if all([s in exp_sensors for s in sensor_names]):
                    relevant_experiment_names.append(exp_name)

        return relevant_experiment_names

    def get_theta_names(self, tex=False):
        """
        Returns the parameter names of the parameter vector theta in the
        corresponding order.

        Parameters
        ----------
        tex : bool, optional
            If True, the TeX-names of the parameters will be returned,
            otherwise the names as they are used in the code will be returned.

        Returns
        -------
        theta_names : list
            List of strings with the parameter names appearing in theta.
        """
        # assemble the parameter's names in the order as they appear in theta
        theta_names = []
        indices = []
        for prm_name, parameter in self._parameters.items():
            if parameter.index is not None:
                indices.append(parameter.index)
                if tex and parameter.tex is not None:
                    theta_names.append(parameter.tex)
                else:
                    theta_names.append(prm_name)
        # order the theta_names according to their index-values; note that this
        # step is not necessary for insertion ordered dicts (Python 3.6+), since
        # in this case theta_names will already be in the right order
        theta_names = [name for _, name in sorted(zip(indices, theta_names))]
        return theta_names

    def theta_explanation(self, print_it=True, check_consistency=True):
        """
        Prints out or returns a string on how the theta-vector, which is the
        numeric parameter vector that is given to the self.loglike and
        self.logprior methods, is interpreted with respect to the problem's
        parameters. The printout will tell you which parameter is connected to
        which index of theta.

        Parameters
        ----------
        print_it : bool, optional
            If True, the explanation string is printed and not returned. If set
            to False, the info-string is not printed but returned.
        check_consistency : bool, optional
            When True, a consistency check is performed before printing the
            explanations on theta. When False, this check is skipped.

        Returns
        -------
        s : str or None
            The constructed string when 'print_it' was set to False.
        """

        # an explanation is not printed if the problem is inconsistent
        if check_consistency:
            self.check_problem_consistency()

        # collect the list of theta names in the right order
        theta_names = self.get_theta_names()

        # construct the info-string
        s = "\n---------------------\n"
        s += "| Theta | Parameter |\n"
        s += "| index |   name    |\n"
        s += "|-------------------|\n"
        for i, prm_name in enumerate(theta_names):
            s += f"|{i:5d} --> {prm_name:<9s}|\n"
        s += "---------------------\n"

        # print or return s
        if print_it:
            print(s)
        else:
            return s

    def add_forward_model(self, name, forward_model):
        """
        Adds a forward model to the inference problem. Note that multiple
        forward models can be added to one problem.

        Parameters
        ----------
        name : str
            The name of the forward model to be added.
        forward_model : obj[ForwardModelTemplate]
            Defines the forward model. Check out forward_model.py to see a
            template for the forward model definition. The user will then have
            to derive his own forward model from that base class.
        """

        # check if all given model parameters have already been added to the
        # inference problem; note that the forward model can only be added to
        # the problem after the corresponding parameters were defined
        for prm_name in forward_model.prms_def:
            self.check_if_parameter_exists(prm_name)

        # check if the given name for the forward model has already been used
        if name in [*self._forward_models.keys()]:
            raise RuntimeError(
                f"The given name '{name}' for the forward model has already "
                f"been used for another forward model. Please choose another "
                f"name.")

        # check if the given forward model has an output sensor with a name that
        # is already used for an output sensor of another forward model
        for existing_name, existing_fwd_model in self._forward_models.items():
            for output_sensor in existing_fwd_model.output_sensor_names:
                if output_sensor in forward_model.output_sensor_names:
                    raise RuntimeError(
                        f"The given forward model '{name}' has an output "
                        f"sensor '{output_sensor}', \nwhich is also defined as "
                        f"an output sensor in the already defined forward "
                        f"model '{existing_name}'.\nPlease choose a different "
                        f"name for output sensor '{output_sensor}' in forward "
                        f"model '{name}'.")

        # add the given forward model to the internal forward model dictionary
        # under the given forward model name
        self._forward_models[name] = forward_model

    def evaluate_model_response(self, theta, experiment_names=None):
        """
        Evaluates the model response for each forward model for the given
        parameter vector theta and the given experiments.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the model responses should be evaluated.
            Which parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        experiment_names : str, list[str] or None, optional
            Contains the names of all or some of the experiments added to the
            inference  problem. If this argument is None (which is a common use
            case) then all experiments defined in the problem (self.experiments)
            are used. The names provided here define the experiments that the
            forward model is evaluated for.

        Returns
        -------
        model_response_dict : dict
            The first key is the name of the experiment. The values are dicts
            which contain the forward model's output sensor's names as keys
            have the corresponding model responses as values.
        """

        # if experiments is not further specified all experiments added to the
        # problem will be accounted for when computing the model error
        if experiment_names is None:
            experiment_names = [*self._experiments.keys()]
        else:
            # make sure that a given string is converted into a list
            experiment_names = make_list(experiment_names)

        # first, loop over all forward models, and then, over all experiments
        # that are associated with the corresponding model
        model_response_dict = {}
        for fwd_name, forward_model in self._forward_models.items():
            # get the model parameters for the considered forward model
            prms_model = self.get_parameters(theta, forward_model.prms_def)
            # get all experiments referring to the considered forward model
            relevant_experiment_names = self.get_experiment_names(
                forward_model_names=fwd_name, experiment_names=experiment_names)
            # evaluate the forward model for each relevant experiment
            for exp_name in relevant_experiment_names:
                exp_dict = self._experiments[exp_name]
                # prepare the model input values from the experimental data
                sensor_values = exp_dict['sensor_values']
                exp_inp = {input_sensor.name: sensor_values[input_sensor.name]
                           for input_sensor in forward_model.input_sensors}
                inp = {**exp_inp, **prms_model}  # adds the two dictionaries
                # finally, evaluate the forward model for this experiment
                model_response_dict[exp_name] = forward_model(inp)

        return model_response_dict

    def add_noise_model(self, noise_model):
        """
        Adds a noise model to the inference problem.

        Parameters
        ----------
        noise_model : obj[NoiseModelTemplate]
            The noise model object, e.g. from NormalNoise. Check out noise.py to
            see some noise model classes.
        """

        # check if all given noise model parameters have already been added to
        # the inference problem
        for prm_name in noise_model.prms_def:
            if prm_name not in self._parameters.keys():
                raise RuntimeError(
                    f"The noise model parameter '{prm_name}' has not been " +
                    f"defined yet.\nYou have to add all noise model " +
                    f"parameters to the problem before adding the noise " +
                    f"model.\nYou can use the 'add_parameter' method for " +
                    f"this purpose."
                )

        # add the given noise model to the internal noise model dictionary under
        # a name derived from the noise model's sensor names
        noise_model_name = '_'.join(make_list(noise_model.sensors))
        if noise_model_name in [*self._noise_models.keys()]:
            sensors = simplified_list_string(noise_model.sensors)
            raise RuntimeError(
                f"A noise model with the sensors {sensors} has already been "
                f"defined in this problem!")
        self._noise_models[noise_model_name] = noise_model

    def assign_experiments_to_noise_models(self):
        """
        Assigns each noise model the corresponding experiment names, based on
        the sensor names, that are defined for each noise model. This function
        is intended to be called after the problem was fully defined.
        """
        n_experiments_defined = len(self._experiments)
        n_experiments_noise = 0
        for noise_model in self._noise_models.values():
            # get the experiments that contain all of the noise model's sensors
            experiment_names = self.get_experiment_names(
                sensor_names=noise_model.sensors)
            n_experiments_noise += len(experiment_names)
            # add the relevant experiment names to the noise model
            noise_model.add_experiment_names(experiment_names,
                                             self._experiments)

        # check if there is the same number of experiments over all noise models
        # as defined for the inference problem
        if n_experiments_noise != n_experiments_defined:
            # this is not necessarily an error; it also happens in a valid setup
            # when more than one noise model are defined for one forward model;
            # in a future version, there could be an info message here
            pass

        # check that each globally defined experiment appears in one of the
        # noise models
        for exp_name in self._experiments.keys():
            found_it = False
            if not found_it:
                for noise_model in self._noise_models.values():
                    if not found_it:
                        for exp_name_noise in noise_model.experiment_names:
                            if exp_name == exp_name_noise:
                                found_it = True
                                break
            if not found_it:
                # one may argue, that this could also be only a warning here
                raise RuntimeError(
                    f"The globally defined experiment '{exp_name}' does not "
                    f"appear in any of the noise models!")
