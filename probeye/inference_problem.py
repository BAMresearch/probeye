# standard library imports
import copy

# third party imports
import numpy as np
from tabulate import tabulate

# local imports
from probeye.parameter import Parameters, ParameterProperties
from probeye.priors import PriorNormal
from probeye.priors import PriorLognormal
from probeye.priors import PriorUniform
from probeye.priors import PriorWeibull
from probeye.subroutines import underlined_string, titled_table
from probeye.subroutines import simplified_list_string, simplified_dict_string
from probeye.subroutines import unvectorize_dict_values


class InferenceProblem:
    """
    This class provides a general framework for defining an inference problem.
    Capabilities for solving the set up problem are intentionally not included.
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : string
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
        # experiments (see self.add_experiment); this dict is managed
        # internally and should not be edited directly
        self._experiments = {}

        # here, the available prior classes are stored; additional ones can be
        # added via self.add_prior_class; see also priors.py; this dictionary is
        # managed internally and should not be edited directly
        self._prior_classes = {'normal': PriorNormal,
                               'lognormal': PriorLognormal,
                               'uniform': PriorUniform,
                               'weibull': PriorWeibull}

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
    def n_calibration_prms(self):
        """Provides n_calibration_prms attribute."""
        return self._parameters.n_calibration_prms

    @property
    def n_constant_prms(self):
        """Provides n_constant_prms attribute."""
        return self._parameters.n_constant_prms

    def info(self, print_it=True, include_experiments=False, tablefmt="presto"):
        """
        Either prints the problem definition to the console (print_it=True) or
        just returns the generated string without printing it (print_it=False).

        Parameters
        ----------
        print_it : boolean, optional
            If True, the generated string is printed and not returned. If set
            to False, the generated string is not printed but returned.
        include_experiments : boolean, optional
            If True, information on the experiments defined within the model
            will be included in the printout. Depending on the number of defined
            experiments, this might result in a long additional printout, which
            is why this is set to False (no experiment printout) by default.
        tablefmt : string
            An argument for the tabulate function defining the style of the
            generated table. Check out tabulate's documentation for more info.

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
        theta_string += self.theta_explanation(print_it=False)

        # print information on added experiments if requested
        if include_experiments:
            rows = []
            for name, exp_dict in self._experiments.items():
                dict_atoms = unvectorize_dict_values(exp_dict['sensors'])
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
        prm_name : string
            The name of the parameter which should be added to the problem.
        prm_type : string
            Either 'model' (for a model parameter), 'prior' (for a prior
            parameter) or 'noise' (for a noise parameter). Note that 'prior'
            parameter do not have to be added by hand. They are added
            internally when 'calibration'-parameters are added.
        const : float or None, optional
            If the added parameter is a 'const'-parameter, the corresponding
            value has to be specified by this argument.
        prior : tuple of two elements or None, optional
            If the added parameter is a 'calibration'-parameter, this argument
            has to be given as a 2-tuple. The first element (a string) defines
            the prior-type (must be a key of self._prior_classes). The second
            element must be a dictionary stating the prior's parameters as keys
            and their numeric values as values. An example for a normal prior:
            ('normal', {'loc': 0.0, 'scale': 1.0}). In order to define the
            prior's parameters, check out the prior definitions in priors.py.
        info : string, optional
            Short explanation on the added parameter.
        tex : string or None, optional
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
            prior_type = prior[0]  # e.g. 'normal', 'lognormal', etc.
            prior_dict = prior[1]  # dictionary with parameter-value pairs
            prior_parameter_names = []
            for prior_parameter_name, value in prior_dict.items():
                # create unique name for this prior parameter
                new_name = f"{prior_parameter_name}_{prm_name}"
                prior_parameter_names.append(new_name)
                # the prior parameter is considered a 'const'-parameter and
                # added to the problem accordingly here
                default_info = f"{prior_type.capitalize()} prior's parameter "
                default_info += f"for calibration-parameter '{prm_name}'"
                self.add_parameter(new_name, 'prior', const=value,
                                   info=default_info)  # recursive call
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

    def check_if_parameter_exists(self, prm_name):
        """
        Checks if a parameter, given by its name, exists within the problem.

        Parameters
        ----------
        prm_name : string
            A global parameter name.
        """
        # check if the given parameter exists
        if prm_name not in self._parameters.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet.")

    def remove_parameter(self, prm_name):
        """
        Removes a parameter ('const' or 'calibration') from inference problem.

        Parameters
        ----------
        prm_name : string
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
            del self._parameters[prm_name].prior
            del self._parameters[prm_name]
            # correct the indices of the remaining 'calibration'-parameters
            idx = 0
            for name, parameter in self._parameters.items():
                if parameter.index is not None:
                    parameter.index = idx
                    idx += 1

    def change_parameter_role(self, prm_name, const=None, prior=None,
                              new_info=None, new_tex=None):
        """
        Performs the necessary tasks to change a parameter's role in the problem
        definition. A parameter's role can either be changed from 'const' to
        'calibration' or from 'calibration' to 'const'.

        Parameters
        ----------
        prm_name : string
            The name of the parameter whose role should be changed.
        const : float or None, optional
            If the new role is 'const', the corresponding value has to be
            specified by this argument.
        prior : tuple of two elements or None, optional
            If a 'const'-parameter should be changed to a 'calibration'
            parameter, this argument has to be given as a 2-tuple. The first
            element (a string) defines the prior-type (must be a key of
            self._prior_classes). The second element must be a dictionary
            stating the prior's parameters as keys and their values as values.
            For example: ('normal', {'loc': 0.0, 'scale': 1.0}). In order to
            define the prior's parameters, check out the prior definitions in
            the script priors.py.
        new_info : string or None, optional
            The new string for the explanation of parameter prm_name.
        new_tex : string or None, optional
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
        prm_name : string
            The name of the parameter whose info-string should be changed.
        new_info : string
            The new string for the explanation of parameter prm_name.
        new_tex : string or None
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
        prm_name : string
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
        Adds a prior-object of a calibration parameter to the internal prior
        dictionary.

        Parameters
        ----------
        name : string
            Unique name of the prior. Usually this name has the structure
            <ref_prm>_<prior_type>.
        prior_type : string
            Defines the prior type. Must appear in the keys of the internal
            dictionary self._prior_classes.
        prms_def : list[str]
            States the prior's parameter names.
        ref_prm : string
            The name of the problem's calibration parameter the prior refers to
            (a prior is always defined for a specific calibration parameter).

        Returns
        -------
        obj[PriorTemplate]
            The instantiated prior-object which is also written to the internal
            prior dictionary self._priors.
        """
        prior_class = self._prior_classes[prior_type]
        self._priors[name] = copy.deepcopy(prior_class(ref_prm, prms_def, name))
        return self._priors[name]

    def add_prior_class(self, name, prior_class):
        """
        This method allows to add a user-defined prior-class to the problem, so
        it can be used in the parameter definitions.

        Parameters
        ----------
        name : string
            Name of the prior class to be added.
        prior_class : class
            The user-defined prior-class. See priors.py for examples.
        """
        # add the class to the internal prior-class dictionary if the given
        # name has not been used for one of the existing classes
        if name in self._prior_classes.keys():
            raise RuntimeError(
                f"A prior class with name '{name}' already exists. " +
                f"Please choose another name."
            )
        self._prior_classes[name] = prior_class

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
        assert self._experiments, "No experiments have been defined yet!"
        assert self._noise_models, "No noise models have been defined yet!"

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

    def add_experiment(self, exp_name, sensors=None, fwd_model_name=None):
        """
        Adds a single experiment to the inference problem. Here, an experiment
        is defined as one or more sensors (note that the experiment does not
        distinguish between input and output sensors) which provide some
        measured data. Additionally, a reference to one of the problem's forward
        models is associated with an experiment.

        Parameters
        ----------
        exp_name : string
            The name of the experiment, e.g. "Exp_20May.12". If an experiment
            with a similar name has already been added, it will be overwritten
            and a warning will be thrown.
        sensors : dict
            The keys are the sensor's names, the values are the measured values.
        fwd_model_name : string
            Name of the forward model this experiment refers to.
        """

        # check all keyword arguments are given
        if sensors is None:
            raise RuntimeError(
                f"No sensors given!"
            )
        if fwd_model_name is None:
            raise RuntimeError(
                f"No forward-model name given!"
            )

        # check if the given forward model exists
        if fwd_model_name not in self._forward_models.keys():
            raise RuntimeError(
                f"The forward model '{fwd_model_name}' does not exist! You "
                f"need to define it before adding experiments that refer to it."
            )

        # throw warning when the experiment name was defined before
        if exp_name in self._experiments.keys():
            print(f"WARNING - Experiment '{exp_name}' is already defined" +
                  f" and will be overwritten!")

        # add the experiment to the central dictionary
        self._experiments[exp_name] = {'sensors': sensors,
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

    def get_experiments(self, forward_model_name, experiments=None):
        """
        Extracts all experiments which refer to a given forward model from a
        given dictionary of experiments.

        Parameters
        ----------
        forward_model_name : string
            The name of the forward model the experiments should refer to.
        experiments : dict or None, optional
            The experiments to search in. If None, all experiments defined
            within the problem will be searched. If a dictionary is given, it
            must have a similar structure as self._experiments.

        Returns
        -------
        relevant_experiments : dict
            The keys are the experiment names, and the values are the 'sensors'
            dictionaries as defined by self.add_experiment.
        """

        # if experiments is not further specified it is assumed that all given
        # experiments should be used
        if experiments is None:
            experiments = self._experiments

        # get the experiments which refer to the given forward model
        relevant_experiments = {}
        for exp_name, experiment in experiments.items():
            if experiment['forward_model'] == forward_model_name:
                relevant_experiments[exp_name] = experiment['sensors']

        return relevant_experiments

    def get_theta_names(self, tex=False):
        """
        Returns the parameter names of the parameter vector theta in the
        corresponding order.

        Parameters
        ----------
        tex : boolean, optional
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

    def theta_explanation(self, print_it=True):
        """
        Prints out or returns a string on how the theta-vector, which is the
        numeric parameter vector that is given to the self.loglike and
        self.logprior methods, is interpreted with respect to the problem's
        parameters. The printout will tell you which parameter is connected to
        which index of theta.

        Parameters
        ----------
        print_it : boolean, optional
            If True, the explanation string is printed and not returned. If set
            to False, the info-string is not printed but returned.

        Returns
        -------
        s : string or None
            The constructed string when 'print_it' was set to False.
        """

        # an explanation is not printed if the problem is inconsistent
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
        name : string
            The name of the forward model to be added.
        forward_model : obj[ModelTemplate]
            Defines the forward model. Check out forward_model.py to see a
            template for the forward model definition. The user will then have
            to derive his own forward model from that base class.
        """

        # check if all given model parameters have already been added to the
        # inference problem; note that the forward model can only be added to
        # the problem after the corresponding parameters were defined
        for prm_name in forward_model.prms_def:
            if prm_name not in self._parameters.keys():
                raise RuntimeError(
                    f"The model parameter '{prm_name}' has not been defined " +
                    f"yet.\nYou have to add all model parameters to the " +
                    f"problem before adding the model.\nYou can use the " +
                    f"'add_parameter' method for this purpose."
                )

        # add the given forward model to the internal forward model dictionary
        # under the given forward model name
        self._forward_models[name] = forward_model

    def evaluate_model_error(self, theta, experiments=None):
        """
        Evaluates the model error for the given parameter vector and the given
        experiments.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the model error should be evaluated.
            Which parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        experiments : dict or None, optional
            Contains all or some of the experiments added to the inference
            problem. Check out the self.add_experiment method to see how this
            dictionary must be structured. If this argument is None (which is
            the common use case) then all experiments defined in the problem
            (self.experiments) are used.

        Returns
        -------
        model_error_dict : dict
            The first key is the name of the corresponding forward model. The
            values are dictionaries which contain the problem's output sensor
            names as keys, and have the corresponding model errors as values.
        """

        # if experiments is not further specified all experiments added to the
        # problem will be accounted for when computing the model error
        if experiments is None:
            experiments = self._experiments

        # the model error is computed within the model
        model_error_dict = {}
        for fwd_name, forward_model in self._forward_models.items():
            prms_model = self.get_parameters(theta, forward_model.prms_def)
            relevant_experiments = self.get_experiments(
                fwd_name, experiments=experiments)
            model_error_dict[fwd_name] = forward_model.error(
                prms_model, relevant_experiments)

        return model_error_dict

    def add_noise_model(self, output_sensor_name, noise_model):
        """
        Adds an output-sensor-specific noise model to the inference problem.

        Parameters
        ----------
        output_sensor_name : string
            The name of the output sensor(s) the noise model refers to, for
            example 'ForceSensor' or 'Clock_1'.
        noise_model : obj[NoiseTemplate]
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
        # the given name of the output sensor(s)
        self._noise_models[output_sensor_name] = noise_model

    def logprior(self, theta):
        """
        Evaluates the log-prior function of the problem at theta.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the log-likelihood function should be
            evaluated. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.

        Returns
        -------
        lp : float
            The evaluated log-prior function for the given theta-vector.
        """
        lp = 0.0
        for prior in self._priors.values():
            prms = self.get_parameters(theta, prior.prms_def)
            lp += prior(prms, 'logpdf')
        return lp

    def sample_from_prior(self, prm_name, size):
        """
        Generates random samples from a parameter's prior distribution and
        returns the generated samples.

        Parameters
        ----------
        prm_name : string
            The name of the parameter the prior is associated with.
        size : int
            The number of random samples to be drawn.

        Returns
        -------
        numpy.ndarray
            The generated samples.
        """
        prior = self._priors[self._parameters[prm_name].prior.name]
        # check for prior-priors; if a prior parameter is a calibration
        # parameter and not a constant, one first samples from the prior
        # parameter's prior distribution, and then takes the mean of those
        # samples to sample from the first prior distribution; this procedure
        # is recursive, so that (in principle) one could also define priors of
        # the prior's prior parameters and so forth
        theta_aux = [0] * self._parameters.n_calibration_prms
        for prior_prm_name in prior.prms_def_no_ref.keys():
            if self._parameters[prior_prm_name].role == 'calibration':
                samples = self.sample_from_prior(prior_prm_name, size)
                theta_aux[self._parameters[prior_prm_name].index] =\
                    np.mean(samples)
        prms = self.get_parameters(theta_aux, prior.prms_def_no_ref)
        return prior.generate_samples(prms, size)

    def loglike(self, theta):
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the log-likelihood function should be
            evaluated. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.

        Returns
        -------
        ll : float
            The evaluated log-likelihood function for the given theta-vector.
        """

        # evaluate the model error for each defined forward model and each
        # output sensor in this/those forward model(s)
        model_error_dict = self.evaluate_model_error(theta)

        # compute the contribution to the log-likelihood function for the
        # model error of forward model and output sensor, and sum it all up
        ll = 0.0
        for me_dict in model_error_dict.values():
            for sensor, me_vector in me_dict.items():
                noise_model = self._noise_models[sensor]
                prms_noise = self.get_parameters(theta, noise_model.prms_def)
                ll += noise_model.loglike_contribution(me_vector, prms_noise)

        return ll
