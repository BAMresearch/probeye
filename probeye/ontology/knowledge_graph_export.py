# standard library
import os

# third party imports
import numpy as np
import arviz as az
from owlready2 import World

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.subroutines import make_list, len_or_one
from probeye.subroutines import add_index_to_tex_prm_name


def load_owl_file(owl_basename):
    # get the path of the owl-file (it is stored in the probeye directory) and load it
    owl_dir = os.path.dirname(__file__)
    owl_file = os.path.join(owl_dir, owl_basename)
    assert os.path.isfile(owl_file), f"Could not find the owl-file at '{owl_file}'"
    peo = World().get_ontology(owl_file).load()
    return peo


def add(subject, property_, object_):
    """
    Adds a triple to the graph after checking if the given property exists.
    """
    assert hasattr(subject, property_)
    # the last element of the triple is always a list; if this list is empty (this
    # is checked by the first 'if') it is added as a single-element-list; if this
    # list already has elements, the given object is merely appended to that list
    if not getattr(subject, property_):
        setattr(subject, property_, make_list(object_))
    else:
        getattr(subject, property_).append(object_)


def export_knowledge_graph(
    problem: InverseProblem,
    output_file: str,
    data_dir: str,
    owl_basename: str = "parameter_estimation_ontology.owl",
):
    """
    Exports a given InferenceProblem to an rdf-file according to the referenced
    parameter estimation ontology.

    Parameter
    ---------
    problem
        The InverseProblem that should be exported to an rdf-file.
    output_file
        Path to the file the knowledge graph should be written to.
    owl_basename
        The basename plus extension of the owl-file that contains the parameter
        estimation ontology. This file must be contained in the probeye directory one
        level above the directory of this file.
    """

    # load the given ontology
    peo = load_owl_file(owl_basename)

    def set_latent_or_const_parameter(subject, property_, const_or_latent):
        """
        Adds a triple to the graph after checking if the given property exists and after
        checking if the given object is a constant of a latent parameter.
        """
        assert hasattr(subject, property_)
        if problem.parameters[const_or_latent].is_const:
            setattr(subject, property_, make_list(peo.constant(const_or_latent)))
        else:
            setattr(subject, property_, make_list(peo.parameter(const_or_latent)))

    def append_latent_or_const_parameter(list_, const_or_latent_name):
        """
        Adds either a constant or a latent parameter to a given collecting list.
        """
        if problem.parameters[const_or_latent_name].is_const:
            list_.append(peo.constant(const_or_latent_name))
        else:
            list_.append(peo.parameter(const_or_latent_name))

    # -------------------------------------------------------------------------------- #
    #                  Add the INVERSE PROBLEM to the knowledge graph                  #
    # -------------------------------------------------------------------------------- #

    # this instance is created to have a common graph root for the instances to come;
    # note that white-space is not allowed in the name, so it is replaced
    inverse_problem = peo.inverse_problem(problem.name.replace(" ", "_"))

    # -------------------------------------------------------------------------------- #
    #                Add the problem's CONSTANTS to the knowledge graph                #
    # -------------------------------------------------------------------------------- #

    for const_name in problem.constant_prms:

        # add the constant with some basic properties
        constant = peo.constant(const_name)
        add(constant, "has_dimension", problem.parameters[const_name].dim)
        if problem.parameters[const_name].dim == 1:
            add(constant, "has_scalar_value", problem.parameters[const_name].value)
        else:
            data = problem.parameters[const_name].value
            filename = os.path.join(data_dir, f"{const_name}.dat")
            np.savetxt(filename, data)
            add(constant, "has_file", filename)
        if problem.parameters[const_name].info is not None:
            add(constant, "has_explanation", problem.parameters[const_name].info)
        if problem.parameters[const_name].tex is not None:
            add(constant, "has_tex_symbol", problem.parameters[const_name].tex)
        add(inverse_problem, "has_constant", constant)

    # -------------------------------------------------------------------------------- #
    #               Add the problem's PARAMETERS to the knowledge graph                #
    # -------------------------------------------------------------------------------- #

    for prm_name in problem.latent_prms:

        # add the parameter with some basic properties
        parameter = peo.parameter(prm_name)
        add(parameter, "has_dimension", problem.parameters[prm_name].dim)
        if problem.parameters[prm_name].info is not None:
            add(parameter, "has_explanation", problem.parameters[prm_name].info)
        if problem.parameters[prm_name].tex is not None:
            add(parameter, "has_tex_symbol", problem.parameters[prm_name].tex)

        # assign the parameter's domain based on its dimension
        domain_name = f"domain_{prm_name}"
        for i, domain_obj in enumerate(problem.parameters[prm_name].domain):
            # the following distinction accounts for the possibility of parameters
            # being multidimensional (only those need their domains to have indices)
            if problem.parameters[prm_name].dim > 1:
                domain_1d = peo.one_dimensional_interval(f"{domain_name}_{i}")
                add(domain_1d, "has_index", i)
            else:
                domain_1d = peo.one_dimensional_interval(domain_name)
            add(domain_1d, "has_lower_bound_value", domain_obj.lower_bound)
            add(domain_1d, "lower_bound_included", domain_obj.lower_bound_included)
            add(domain_1d, "has_upper_bound_value", domain_obj.upper_bound)
            add(domain_1d, "upper_bound_included", domain_obj.upper_bound_included)
            add(parameter, "has_domain", domain_1d)

        # assign the parameter's prior
        prior_name = problem.parameters[prm_name].prior.name
        prior_type = problem.parameters[prm_name].prior.prior_type

        if prior_type == "normal":
            prior = peo.normal_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)
            mean_name = f"mean_{prm_name}"
            set_latent_or_const_parameter(prior, "has_mean", mean_name)
            std_name = f"std_{prm_name}"
            set_latent_or_const_parameter(prior, "has_standard_deviation", std_name)

        elif prior_type == "multivariate-normal":
            prior = peo.normal_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)
            mean_name = f"mean_{prm_name}"
            set_latent_or_const_parameter(prior, "has_mean", mean_name)
            cov_name = f"cov_{prm_name}"
            set_latent_or_const_parameter(prior, "has_covariance_matrix", cov_name)

        elif prior_type == "truncnormal":
            prior = peo.truncated_normal_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)
            mean_name = f"mean_{prm_name}"
            set_latent_or_const_parameter(prior, "has_mean", mean_name)
            std_name = f"std_{prm_name}"
            set_latent_or_const_parameter(prior, "has_standard_deviation", std_name)
            a_name = f"a_{prm_name}"
            set_latent_or_const_parameter(prior, "has_lower_bound", a_name)
            b_name = f"b_{prm_name}"
            set_latent_or_const_parameter(prior, "has_upper_bound", b_name)

        elif prior_type == "lognormal":
            prior = peo.lognormal_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)
            mean_name = f"mean_{prm_name}"
            set_latent_or_const_parameter(prior, "has_mean", mean_name)
            std_name = f"std_{prm_name}"
            set_latent_or_const_parameter(prior, "has_standard_deviation", std_name)

        elif prior_type == "uniform":
            prior = peo.uniform_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)
            low_name = f"low_{prm_name}"
            set_latent_or_const_parameter(prior, "has_lower_bound", low_name)
            high_name = f"high_{prm_name}"
            set_latent_or_const_parameter(prior, "has_upper_bound", high_name)

        elif prior_type == "sample-based":
            prior = peo.sample_based_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)
            add(prior, "has_samples", peo.constant(f"samples_{prm_name}"))

        elif prior_type == "uninformative":
            prior = peo.uninformative_density_function(prior_name)
            add(prior, "has_primary_variable", parameter)

        else:
            raise RuntimeError(
                f"Encountered unknown prior type '{prior_type}'. Currently no routines "
                f"for this prior's knowledge graph export are implemented."
            )
        add(parameter, "has_prior_distribution", prior)
        add(inverse_problem, "has_latent_parameter", parameter)

    # -------------------------------------------------------------------------------- #
    #             Add the problem's FORWARD MODELS to the knowledge graph              #
    # -------------------------------------------------------------------------------- #

    for fwd_name, fwd_model in problem.forward_models.items():

        forward_model = peo.forward_model(fwd_name)
        namespace = peo.get_namespace(fwd_name)

        # add the forward model's parameters; the following structure accounts for the
        # fact that a forward model's parameters can be either latent or constant
        const_and_latent = []  # type: list
        for prm_name in fwd_model.prms_def:
            append_latent_or_const_parameter(const_and_latent, prm_name)
        add(forward_model, "has_parameter", const_and_latent)

        # add the forward model's input sensors
        input_sensors = []
        for sensor_obj in fwd_model.input_sensors:
            input_sensors.append(peo.input_sensor(sensor_obj.name, namespace=namespace))
        add(forward_model, "has_input_sensor", input_sensors)

        # add the forward model's output sensors
        output_sensors = []
        for sensor_obj in fwd_model.output_sensors:
            sensor = peo.output_sensor(sensor_obj.name, namespace=namespace)
            for exp_name, measured_values in sensor_obj.items():
                c = peo.constant(sensor_obj.name, namespace=peo.get_namespace(exp_name))
                add(sensor, "has_measured_values", c)
            set_latent_or_const_parameter(
                sensor, "model_prediction_error_described_by", sensor_obj.std_model
            )
            output_sensors.append(sensor)
        add(forward_model, "has_output_sensor", output_sensors)
        add(inverse_problem, "has_forward_model", forward_model)

    # -------------------------------------------------------------------------------- #
    #               Add the problem's EXPERIMENTS to the knowledge graph               #
    # -------------------------------------------------------------------------------- #

    for exp_name, exp_dict in problem.experiments.items():

        # this is where the experiment instance is added to the graph
        experiment = peo.single_experiment_data_set(exp_name)
        namespace = peo.get_namespace(exp_name)

        # associate the experiment's forward model and its sensors
        add(experiment, "is_modeled_by", peo.forward_model(exp_dict["forward_model"]))
        fwd_namespace = peo.get_namespace(exp_dict["forward_model"])
        sensors = []
        for isensor in problem.forward_models[exp_dict["forward_model"]].input_sensors:
            input_sensor = peo.input_sensor(isensor.name, namespace=fwd_namespace)
            sensors.append(input_sensor)
        for osensor in problem.forward_models[exp_dict["forward_model"]].output_sensors:
            output_sensor = peo.input_sensor(osensor.name, namespace=fwd_namespace)
            sensors.append(output_sensor)
        add(experiment, "has_sensor", sensors)

        # associate the sensor values with the experiment
        measurements = []
        for sensor_name, sensor_data in exp_dict["sensor_values"].items():
            constant = peo.constant(sensor_name, namespace=namespace)
            if len_or_one(sensor_data) == 1:
                add(constant, "has_scalar_value", sensor_data)
            else:
                filename = os.path.join(data_dir, f"{sensor_name}_{exp_name}.dat")
                filename = os.path.abspath(filename)
                np.savetxt(filename, sensor_data)
                add(constant, "has_file", filename)
            measurements.append(constant)
        add(experiment, "has_measured_values", measurements)
        add(inverse_problem, "has_experiment", experiment)

    # -------------------------------------------------------------------------------- #
    #            Add the problem's LIKELIHOOD MODELS to the knowledge graph            #
    # -------------------------------------------------------------------------------- #

    for i, like_obj in enumerate(problem.likelihood_models.values()):

        # instantiate the data generation model that the likelihood model is based on
        name = f"data_generation_model_{i + 1}"
        data_generation_model = peo.addition(name)
        experiment = peo.single_experiment_data_set(like_obj.experiment_name)
        add(data_generation_model, "describes_experiment", experiment)
        namespace = peo.get_namespace(name)

        # this is just for shorter code
        fwd_model = like_obj.forward_model

        # the data generation model is essentially a mathematical function that adds up
        # different terms; these terms (summands) will be collected in this list
        list_of_summands = []

        if like_obj.additive_model_error:

            # ---------------------------------------- #
            #         Forward model (additive)         #
            # ---------------------------------------- #

            # the first summand is the forward model prediction; since the forward
            # model's output can be returned via multiple output sensors (hence multiple
            # vectors) it might be necessary to concatenate those to a single vector
            if fwd_model.n_output_sensors > 1:

                # in this case, the values returned via the multiple output sensors
                # need to be vectorized before they can be used as a summand
                vec = peo.concatenation("concatenation", namespace=namespace)
                model_vector = peo.variable("model_vector", namespace=namespace)
                add(vec, "returns", model_vector)
                for output_sensor in fwd_model.output_sensors:
                    out = peo.output_sensor(
                        output_sensor.name,
                        namespace=peo.get_namespace(fwd_model.name),
                    )
                    add(vec, "has_argument", out)
                list_of_summands.append(model_vector)

            else:

                # in this case, no concatenation is necessary
                list_of_summands.append(
                    peo.output_sensor(
                        fwd_model.output_sensor.name,
                        namespace=peo.get_namespace(fwd_model.name),
                    )
                )

            # -------------------------------------- #
            #         Model error (additive)         #
            # -------------------------------------- #

            # add the zero-mean normal random variable
            nrv = peo.normal_random_variable("model_error", namespace=namespace)
            add(nrv, "has_mean", peo.zero)

            if like_obj.considers_correlation:

                # in the case correlation is considered, the creation of a correlation
                # matrix is described by a covariance matrix assembler
                cov_matrix = peo.variable("cov_matrix", namespace=namespace)
                cov_assembler = peo.covariance_matrix_assembler(
                    "cov_assembler", namespace=namespace
                )
                add(cov_assembler, "returns", cov_matrix)
                add(nrv, "has_covariance_matrix", cov_matrix)

                # associate the covariance assembler with the respective model error
                # standard deviations which are defined in the forward model
                std_model = []  # type: list
                for output_sensor in like_obj.forward_model.output_sensors:
                    std_model_name = output_sensor.std_model
                    append_latent_or_const_parameter(std_model, std_model_name)
                add(cov_assembler, "has_standard_deviation", std_model)

                # associate the covariance assembler with a correlation function
                if like_obj.correlation_model == "exp":
                    corr_function = peo.correlation_function(
                        "exp_corr_function", namespace=namespace
                    )
                    add(cov_assembler, "uses_function", corr_function)
                    # associate the correlation lengths with this correlation function
                    corr_lengths = []  # type: list
                    for output_sensor in fwd_model.output_sensors:
                        for corr_length in output_sensor.correlated_in.values():
                            append_latent_or_const_parameter(corr_lengths, corr_length)
                    add(corr_function, "has_correlation_length", corr_lengths)

            else:

                # in this case, the Gaussian model error has a diagonal covariance
                # matrix (no correlation) which contains the squared values of (possibly
                # multiple) model error standard deviations; these are collected here
                # and afterwards associated with the random variable "model_error"
                std_model = []
                for output_sensor in like_obj.forward_model.output_sensors:
                    std_model_name = output_sensor.std_model
                    append_latent_or_const_parameter(std_model, std_model_name)
                add(nrv, "has_standard_deviation", std_model)
                list_of_summands.append(nrv)

        else:

            mult = peo.multiplication("multiplication", namespace=namespace)
            product = peo.variable("product", namespace=namespace)
            add(mult, "returns", product)
            list_of_summands.append(product)

            # this is for collecting the two factors of the first summand
            list_of_factors = []

            # ---------------------------------------------- #
            #         Forward model (multiplicative)         #
            # ---------------------------------------------- #

            # the first factor is the forward model prediction; since the forward
            # model's output can be returned via multiple output sensors (hence multiple
            # vectors) it might be necessary to concatenate those to a single vector
            if fwd_model.n_output_sensors > 1:

                # in this case, the values returned via the multiple output sensors
                # need to be vectorized before they can be used as a summand
                vec = peo.concatenation("concatenation", namespace=namespace)
                model_vector = peo.variable("model_vector", namespace=namespace)
                add(vec, "returns", model_vector)
                for output_sensor in fwd_model.output_sensors:
                    out = peo.output_sensor(
                        output_sensor.name,
                        namespace=peo.get_namespace(fwd_model.name),
                    )
                    add(vec, "has_argument", out)
                list_of_factors.append(model_vector)

            else:

                # in this case, no concatenation is necessary
                list_of_factors.append(
                    peo.output_sensor(
                        fwd_model.output_sensor.name,
                        namespace=peo.get_namespace(fwd_model.name),
                    ),
                )

            # ---------------------------------------------- #
            #          Model error (multiplicative)          #
            # ---------------------------------------------- #

            # add the unit-mean normal random variable
            nrv = peo.normal_random_variable("model_error", namespace=namespace)
            add(nrv, "has_mean", peo.one)

            if like_obj.considers_correlation:

                # in the case correlation is considered, the creation of a correlation
                # matrix is described by a covariance matrix assembler
                cov_matrix = peo.variable("cov_matrix", namespace=namespace)
                cov_assembler = peo.covariance_matrix_assembler(
                    "cov_assembler", namespace=namespace
                )
                add(cov_assembler, "returns", cov_matrix)
                add(nrv, "has_covariance_matrix", cov_matrix)

                # associate the covariance assembler with the respective model error
                # standard deviations which are defined in the forward model
                std_model = []
                for output_sensor in like_obj.forward_model.output_sensors:
                    std_model_name = output_sensor.std_model
                    append_latent_or_const_parameter(std_model, std_model_name)
                add(cov_assembler, "has_standard_deviation", std_model)

                # associate the covariance assembler with a correlation function
                if like_obj.correlation_model == "exp":
                    corr_function = peo.correlation_function(
                        "exp_corr_function", namespace=namespace
                    )
                    add(cov_assembler, "uses_function", corr_function)
                    # associate the correlation lengths with this correlation function
                    corr_lengths = []
                    for output_sensor in fwd_model.output_sensors:
                        for corr_length in output_sensor.correlated_in.values():
                            append_latent_or_const_parameter(corr_lengths, corr_length)
                    add(corr_function, "has_correlation_length", corr_lengths)

            else:

                # in this case, the Gaussian model error has a diagonal covariance
                # matrix (no correlation) which contains the squared values of (possibly
                # multiple) model error standard deviations; these are collected here
                # and afterwards associated with the random variable "model_error"
                std_model = []
                for output_sensor in like_obj.forward_model.output_sensors:
                    std_model_name = output_sensor.std_model
                    append_latent_or_const_parameter(std_model, std_model_name)
                add(nrv, "has_standard_deviation", std_model)
                list_of_factors.append(nrv)
            add(mult, "has_factor", list_of_factors)

        # ------------------------------------- #
        #           Measurement error           #
        # ------------------------------------- #

        if like_obj.additive_measurement_error:

            # add the zero-mean normal random variable
            nrv = peo.normal_random_variable("measurement_error", namespace=namespace)
            add(nrv, "has_mean", peo.zero)

            # add all of the contributing measurement error standard deviations
            std_meas = []  # type: list
            for output_sensor in like_obj.forward_model.output_sensors:
                std_meas_name = output_sensor.std_measurement
                append_latent_or_const_parameter(std_meas, std_meas_name)
            add(nrv, "has_standard_deviation", std_meas)
            list_of_summands.append(nrv)

        # add all of the summands
        add(data_generation_model, "has_summand", list_of_summands)

        # add the data generation model to the inverse problem
        add(inverse_problem, "has_data_generation_model", data_generation_model)

    # write the graph to the specified output file
    peo.save(file=output_file)


def export_results_to_knowledge_graph(
    problem: InverseProblem,
    inference_data: az.data.inference_data.InferenceData,
    output_file: str,
    data_dir: str,
):
    """
    Adds the results of a solver to the graph of an inverse problem.

    Parameter
    ---------
    problem
        The InverseProblem that should be exported to an rdf-file.
    inference_data
        The data object returned by one of the solvers.
    output_file
        Path to the file the knowledge graph should be written to.
    owl_basename
        The basename plus extension of the owl-file that contains the parameter
        estimation ontology. This file must be contained in the probeye directory one
        level above the directory of this file.
    """

    # load the given ontology
    peo = load_owl_file(output_file)

    if "posterior" in inference_data:

        for prm_name in problem.latent_prms:

            # assign the posterior to the parameter
            parameter = peo.parameter(prm_name)
            posterior = peo.sample_based_density_function(f"posterior_{prm_name}")
            add(posterior, "has_primary_variable", parameter)
            add(parameter, "has_posterior_distribution", posterior)

            # prepare the constant that contains the sample-data
            samples_name = f"samples_{prm_name}"
            samples = peo.constant(samples_name)
            if problem.parameters[prm_name].dim == 1:
                tex_name = problem.parameters[prm_name].tex
                data = inference_data["posterior"][tex_name].values
                filename = os.path.join(data_dir, f"{samples_name}.dat")
                # noinspection PyTypeChecker
                np.savetxt(filename, data)
                add(samples, "has_file", filename)
            else:
                for i in range(1, problem.parameters[prm_name].dim + 1):
                    tex_name = problem.parameters[prm_name].tex
                    tex_name = add_index_to_tex_prm_name(tex_name, i)
                    data = inference_data["posterior"][tex_name].values
                    filename = os.path.join(data_dir, f"{samples_name}_{i}.dat")
                    # noinspection PyTypeChecker
                    np.savetxt(filename, data)
                    add(samples, "has_file", filename)

            # assign the samples to the posterior distribution
            add(posterior, "has_samples", samples)

    else:

        for prm_name in problem.latent_prms:

            # assign the posterior to the parameter
            parameter = peo.parameter(prm_name)

            # prepare the constant that contains the maximum likelihood estimate
            ml_estimate_name = f"max_likelihood_estimate_{prm_name}"
            ml_estimate = peo.constant(ml_estimate_name)

            # this is the np.ndarray with the maximum likelihood estimate for all
            # latent parameters as a vector
            theta_ml = getattr(inference_data, "x")

            # write the estimate either directly to the constant (in case it's a scalar)
            # or to a file (in case of a multi-dimensional parameter)
            if problem.parameters[prm_name].dim == 1:
                idx = problem.parameters[prm_name].index
                add(ml_estimate, "has_scalar_value", float(theta_ml[idx]))
            else:
                idx_start = problem.parameters[prm_name].index
                idx_end = problem.parameters[prm_name].index_end
                data = theta_ml[idx_start:idx_end]
                filename = os.path.join(data_dir, f"{ml_estimate_name}.dat")
                np.savetxt(filename, data)
                add(ml_estimate, "has_file", filename)

            # make the association to the parameter
            add(parameter, "has_maximum_likelihood_estimate", ml_estimate)

    # write the graph to the specified output file
    peo.save(file=output_file)
