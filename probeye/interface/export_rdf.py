# standard library
import os
from typing import Union

# third party imports
import owlready2
from rdflib import URIRef, Graph, Literal
from rdflib.namespace import RDF, XSD

# local imports
from probeye.definition.inference_problem import InferenceProblem
from probeye.subroutines import iri
from probeye.subroutines import add_constant_to_graph

def export_rdf(
    problem: InferenceProblem,
    ttl_file: str,
    owl_basename: str = "parameter_estimation_ontology.owl",
    include_explanations: bool = True,
):
    """
    Exports a given InferenceProblem to an rdf-file according to the referenced
    parameter estimation ontology.

    Parameter
    ---------
    problem
        The InferenceProblem that should be exported to an rdf-file.
    ttl_file
        Path to the file (turtle-format) the knowledge graph should be written to.
    owl_basename
        The basename plus extension of the owl-file that contains the parameter
        estimation ontology. This file must be contained in the probeye directory one
        level above the directory of this file.
    include_explanations
        If True, some of the graph's instances will have string-attributes which give a
        short explanation on what they are. If False, those explanations will not be
        included. This might be useful for graph-visualizations.
    """

    # get the full path of the owl-file (it is stored in the probeye directory)
    dir_path = os.path.dirname(__file__)
    owl_dir = os.path.join(dir_path, "..")
    owl_file = os.path.join(owl_dir, owl_basename)
    assert os.path.isfile(owl_file), f"Could not find the owl-file at '{owl_file}'"

    # owlready2 stores every triple in a so-called 'World' object which connects to a
    # quadstore which can be written to a file; this is prepared here
    quadstore_file = os.path.join(owl_dir, "peo_quadstore.sqlite3")
    peo_world = owlready2.World(filename=quadstore_file)

    # instantiate a plain knowledge graph; one can also instantiate a graph that is
    # connected to the World object and will hence contain the entire ontology via the
    # command 'peo_world.as_rdflib_graph()'
    graph = Graph()

    # the following command loads the ontology from the given owl-file into the
    # quadstore which is connected to the peo_world object; this means that the
    # following command creates the quadstore-file ('peo_quadstore.sqlite3')
    peo = peo_world.get_ontology(owl_file).load()

    # the following simple function is needed at several locations
    def new_namespace_iri(basename: str) -> str:
        """Returns a new namespace IRI defined with a given basename."""
        return f"{peo.base_iri[:-1]}/{basename}#"

    # the following context manager is required for the following triple generation;
    # as a consequence you cannot add arbitrary triples here, but only triples that are
    # consistent with the parameter estimation ontology
    with peo:

        # ---------------------------------------------------------------------------- #
        #              Add the problem's CONSTANTS to the knowledge graph              #
        # ---------------------------------------------------------------------------- #

        for const_name in problem.constant_prms:

            # since the definition of constant numeric data is required multiple times,
            # the functionality is encapsulated in a subroutine
            const_value = problem.parameters[const_name].value
            info = problem.parameters[const_name].info
            use = problem.parameters[const_name].type
            add_constant_to_graph(
                peo,
                graph,
                const_value,
                const_name,
                use,
                info,
                include_explanations=include_explanations,
            )

        # ---------------------------------------------------------------------------- #
        #             Add the problem's EXPERIMENTS to the knowledge graph             #
        # ---------------------------------------------------------------------------- #

        for exp_name, exp_dict in problem.experiments.items():

            # add the experiment as an instance; note that the notation using the
            # auxiliary variables 't1', 't2' and 't3' was chosen intentionally, since a
            # direct notation where those auxiliary variables are not used looks very
            # ugly due to the long code line which will be broken up
            t1 = iri(peo.single_experiment_measurement_data_set(exp_name))
            t2 = RDF.type
            t3 = iri(
                peo.single_experiment_measurement_data_set
            )  # type: Union[URIRef, Literal]
            graph.add((t1, t2, t3))

            # assign the experiment's forward model; note that it is not a problem that
            # this forward model has not been added to the graph yet
            t1 = iri(peo.single_experiment_measurement_data_set(exp_name))
            t2 = iri(peo.is_modelled_by)
            t3 = iri(peo.forward_model(exp_dict["forward_model"]))
            graph.add((t1, t2, t3))

            # create a namespace for the considered experiment; this is required since
            # different experiments can contain data for sensors with similar names;
            # e.g. 'exp_1' and 'exp_2' can both have sensor values for a sensor 'x'
            exp_namespace_iri = new_namespace_iri(exp_name)
            graph.bind(exp_name, URIRef(exp_namespace_iri))
            namespace = peo.get_namespace(exp_namespace_iri)

            with namespace:
                for sensor_name, sensor_value in exp_dict["sensor_values"].items():
                    # this adds a constant-instance to the graph for this sensor
                    use = "experiment"
                    info = f"data from experiment '{exp_name}'"
                    add_constant_to_graph(
                        peo,
                        graph,
                        sensor_value,
                        sensor_name,
                        use,
                        info,
                        include_explanations=include_explanations,
                    )
                    # assign the added constant to the experiment; note that 't1' is
                    # the experiment-instance from above
                    t2 = iri(peo.has_sensor_value)
                    t3 = iri(peo.constant(sensor_name))
                    graph.add((t1, t2, t3))

        # ---------------------------------------------------------------------------- #
        #             Add the problem's PARAMETERS to the knowledge graph              #
        # ---------------------------------------------------------------------------- #

        for prm_name in problem.latent_prms:

            # instantiate the parameter
            t1 = iri(peo.parameter(prm_name))
            t2 = RDF.type
            t3 = iri(peo.parameter)
            graph.add((t1, t2, t3))

            # -------------------- #
            #        Domain        #
            # -------------------- #

            # since a parameter is a variable, its domain is added here
            for i in range(problem.parameters[prm_name].dim):
                # add the domain instance
                if problem.parameters[prm_name].dim == 1:
                    domain_name = f"domain_{prm_name}"
                else:
                    domain_name = f"domain_{prm_name}_index_{i}"
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = RDF.type
                t3 = iri(peo.one_dimensional_interval)
                graph.add((t1, t2, t3))
                # add lower bound to domain instance
                pair = problem.parameters[prm_name].domain[i]
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.has_value)
                t3 = Literal(pair[0], datatype=XSD.float)
                graph.add((t1, t2, t3))
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.lower_bound_included)
                t3 = Literal(False, datatype=XSD.boolean)
                graph.add((t1, t2, t3))
                # add upper bound to domain instance
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.has_value)
                t3 = Literal(pair[1], datatype=XSD.float)
                graph.add((t1, t2, t3))
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.upper_bound_included)
                t3 = Literal(False, datatype=XSD.boolean)
                graph.add((t1, t2, t3))
                # add index to domain instance
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.has_row_index)
                t3 = Literal(i, datatype=XSD.int)
                graph.add((t1, t2, t3))
                # add the use-string to domain instance
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.used_for)
                t3 = Literal("domain", datatype=XSD.string)
                graph.add((t1, t2, t3))
                # add the info-string to domain instance
                if include_explanations:
                    t1 = iri(peo.one_dimensional_interval(domain_name))
                    t2 = iri(peo.has_explanation)
                    t3 = Literal(
                        f"domain definition (index {i}) of parameter '{prm_name}'",
                        datatype=XSD.string,
                    )
                    graph.add((t1, t2, t3))
            # finally, assign the prepared domain to the parameter
            t1 = iri(peo.parameter(prm_name))
            t2 = iri(peo.has_domain)
            t3 = iri(peo.one_dimensional_interval(domain_name))
            graph.add((t1, t2, t3))

            # -------------------- #
            #        Priors        #
            # -------------------- #

            # add the priors and their parameters
            prm_dim = problem.parameters[prm_name].dim
            prior_name = problem.parameters[prm_name].prior.name
            prior_type = problem.parameters[prm_name].prior.prior_type

            if prior_type == "normal":
                # normal density function instance
                t1 = iri(peo.normal_probability_density_function(prior_name))
                t2 = RDF.type
                t3 = iri(peo.normal_probability_density_function)
                graph.add((t1, t2, t3))
                # associate density function with parameter as prior
                t1 = iri(peo.parameter(prm_name))
                t2 = iri(peo.has_prior_belief_distribution)
                t3 = iri(peo.normal_probability_density_function(prior_name))
                graph.add((t1, t2, t3))
                # primary variable
                t1 = iri(peo.normal_probability_density_function(prior_name))
                t2 = iri(peo.has_primary_variable)
                t3 = iri(peo.parameter(prm_name))
                graph.add((t1, t2, t3))
                # mean value input argument
                t2 = iri(peo.has_mean)
                mean_name = f"loc_{prm_name}"
                if problem.parameters[mean_name].is_latent:
                    t3 = iri(peo.parameter(mean_name))
                else:
                    t3 = iri(peo.constant(mean_name))
                graph.add((t1, t2, t3))
                # standard deviation input argument
                if prm_dim == 1:
                    std_name = f"scale_{prm_name}"
                    t2 = iri(peo.has_standard_deviation)
                    if problem.parameters[std_name].is_latent:
                        t3 = iri(peo.parameter(std_name))
                    else:
                        t3 = iri(peo.constant(std_name))
                    graph.add((t1, t2, t3))
                else:
                    cov_name = f"scale_{prm_name}"
                    t2 = iri(peo.has_covariance_matrix)
                    if problem.parameters[cov_name].is_latent:
                        t3 = iri(peo.parameter(cov_name))
                    else:
                        t3 = iri(peo.constant(cov_name))
                    graph.add((t1, t2, t3))

            elif prior_type == "uniform":
                # uniform density function instance
                t1 = iri(peo.uniform_probability_density_function(prior_name))
                t2 = RDF.type
                t3 = iri(peo.uniform_probability_density_function)
                graph.add((t1, t2, t3))
                # associate density function with parameter as prior
                t1 = iri(peo.parameter(prm_name))
                t2 = iri(peo.has_prior_belief_distribution)
                t3 = iri(peo.uniform_probability_density_function(prior_name))
                graph.add((t1, t2, t3))
                # primary variable
                t1 = iri(peo.uniform_probability_density_function(prior_name))
                t2 = iri(peo.has_primary_variable)
                t3 = iri(peo.parameter(prm_name))
                graph.add((t1, t2, t3))
                # lower bound input argument
                t2 = iri(peo.has_lower_bound)
                low_name = f"low_{prm_name}"
                if problem.parameters[low_name].is_latent:
                    t3 = iri(peo.parameter(low_name))
                else:
                    t3 = iri(peo.constant(low_name))
                graph.add((t1, t2, t3))
                # upper bound input argument
                t2 = iri(peo.has_upper_bound)
                high_name = f"high_{prm_name}"
                if problem.parameters[high_name].is_latent:
                    t3 = iri(peo.parameter(high_name))
                else:
                    t3 = iri(peo.constant(high_name))
                graph.add((t1, t2, t3))

            elif prior_type == "histogram":  # not available in probeye yet!
                # histogram density function instance
                t1 = iri(peo.uniform_probability_density_function(prior_name))
                t2 = RDF.type
                t3 = iri(peo.histogram_probability_density_function)
                graph.add((t1, t2, t3))
                # associate density function with parameter as prior
                t1 = iri(peo.parameter(prm_name))
                t2 = iri(peo.has_prior_belief_distribution)
                t3 = iri(peo.histogram_probability_density_function(prior_name))
                graph.add((t1, t2, t3))
                # primary variable
                t1 = iri(peo.histogram_probability_density_function(prior_name))
                t2 = iri(peo.has_primary_variable)
                t3 = iri(peo.vector(prm_name))
                graph.add((t1, t2, t3))
                # histogram bin mid points input argument
                t1 = iri(peo.histogram_probability_density_function(prior_name))
                t2 = iri(peo.has_has_bin_mids)
                t3 = iri(peo.constant(f"hist_bins_{prm_name}"))
                graph.add((t1, t2, t3))
                # histogram bin values input argument
                t1 = iri(peo.histogram_probability_density_function(prior_name))
                t2 = iri(peo.has_has_bin_values)
                t3 = iri(peo.vector(f"hist_values_{prm_name}"))
                graph.add((t1, t2, t3))

            else:
                raise NotImplementedError(
                    f"No prior export for '{prior_type}'-prior implemented yet!"
                )

        # ---------------------------------------------------------------------------- #
        #           Add the problem's FORWARD MODELS to the knowledge graph            #
        # ---------------------------------------------------------------------------- #

        for fwd_name, fwd_model in problem.forward_models.items():

            # add the forward model as an instance
            t1 = iri(peo.forward_model(fwd_name))
            t2 = RDF.type
            t3 = iri(peo.forward_model)
            graph.add((t1, t2, t3))

            # add the model parameters as input arguments; note that it is not a problem
            # that these parameters have not been added to the graph yet
            for prm_name in fwd_model.prms_def:
                t1 = iri(peo.forward_model(fwd_name))
                t2 = iri(peo.has_parameter)
                t3 = iri(peo.parameter(prm_name))
                graph.add((t1, t2, t3))

            for isensor in fwd_model.input_sensors:
                t1 = iri(peo.forward_model(fwd_name))
                t2 = iri(peo.has_input_argument)
                t3 = iri(peo.variable(isensor.name))
                graph.add((t1, t2, t3))
                # TODO: a variable needs a domain

                # input arguments from the input sensors are always specified by
                # experimental data; this connection is assigned here
                for exp_name, exp_dict in problem.experiments.items():
                    exp_namespace_iri = new_namespace_iri(exp_name)
                    with peo.get_namespace(exp_namespace_iri):
                        t1 = iri(peo.constant(isensor.name))
                    t2 = iri(peo.specifies_variable)
                    t3 = iri(peo.variable(isensor.name))
                    graph.add((t1, t2, t3))

            for osensor in fwd_model.output_sensors:
                # create an instance of a variable (the return value)
                with peo.get_namespace(new_namespace_iri(fwd_name)):
                    t1 = iri(peo.variable(osensor.name))
                t2 = RDF.type
                t3 = iri(peo.variable)
                graph.add((t1, t2, t3))
                # identify the variable created before as a return value
                t1 = iri(peo.forward_model(fwd_name))
                t2 = iri(peo.has_return_value)
                with peo.get_namespace(new_namespace_iri(fwd_name)):
                    t3 = iri(peo.variable(osensor.name))
                graph.add((t1, t2, t3))
                # TODO: a variable needs a domain

        # ---------------------------------------------------------------------------- #
        #            Add the problem's NOISE MODELS to the knowledge graph             #
        # ---------------------------------------------------------------------------- #

        for noise_model in problem.noise_models:
            # add noise model instance
            t1 = iri(peo.forward_error_function(noise_model.name))
            t2 = RDF.type
            t3 = iri(peo.mathematical_function)
            graph.add((t1, t2, t3))
            # add the experiments covered by the forward_error_function
            for exp_name in noise_model.experiment_names:
                t2 = iri(peo.describes_error_with_respect_to_experiment)
                t3 = iri(peo.single_experiment_measurement_data_set(exp_name))
                graph.add((t1, t2, t3))
            fwd_model_name = problem.experiments[exp_name]['forward_model']
            t1 = iri(peo.forward_error_function(noise_model.name))
            t2 = iri(peo.has_input_argument)
            for sensor_name in noise_model.sensor_names:
                with peo.get_namespace(new_namespace_iri(fwd_model_name)):
                    t3 = iri(peo.variable(sensor_name))
                graph.add((t1, t2, t3))

            # # add the underlying distribution type
            # if noise_model.dist == "normal":
            #     t1 = iri(peo.forward_error_model(noise_model.name))
            #     t2 = iri(peo.has_base_distribution)
            #     t3 = iri(peo.normal_probability_density_function)
            #     graph.add((t1, t2, t3))
            #
            # # add the noise model's parameters
            # for prm_name in noise_model.prms_def:
            #     t2 = iri(peo.has_input_argument)
            #     t3 = iri(peo.parameter(prm_name))
            #     graph.add((t1, t2, t3))

    # print the triples to a file
    graph.serialize(destination=ttl_file, format="turtle")
