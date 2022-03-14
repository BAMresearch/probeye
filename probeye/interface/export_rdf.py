# standard library
import os
from typing import Union, Optional
import urllib

# third party imports
import numpy as np
import owlready2
import rdflib
from rdflib import URIRef, Graph, Literal
from rdflib.namespace import RDF, XSD

# local imports
from probeye.definition.inference_problem import InferenceProblem
from probeye.subroutines import get_global_name


def export_rdf(
    problem: InferenceProblem,
    ttl_file: str,
    owl_basename: str = "parameter_estimation_ontology.owl",
    quadstore_file: Optional[str] = None,
    part_of_iri: str = "http://www.obofoundry.org/ro/#OBO_REL:part_of",
    has_part_iri: str = "http://www.obofoundry.org/ro/#OBO_REL:has_part",
    include_explanations: bool = False,
    write_array_data: bool = False,
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
    quadstore_file
        Path to the quadstore-file used by owlready2.
    part_of_iri
        The IRI used for the BFO object relation 'part_of'.
    has_part_iri
        The IRI used for the BFO object relation 'has_part' (inverse of part_of).
    include_explanations
        If True, some of the graph's instances will have string-attributes which give a
        short explanation on what they are. If False, those explanations will not be
        included. This might be useful for graph-visualizations.
    write_array_data
        When True, the values and indices of an array are written to the graph. However,
        this might lead to a rather large graph. When False, no values and indices are
        written to the graph. This might be useful when the non-data part of the graph
        is of primary interest.
    """

    # get the full path of the owl-file (it is stored in the probeye directory)
    dir_path = os.path.dirname(__file__)
    owl_dir = os.path.join(dir_path, "..")
    owl_file = os.path.join(owl_dir, owl_basename)
    assert os.path.isfile(owl_file), f"Could not find the owl-file at '{owl_file}'"

    # owlready2 stores every triple in a so-called 'World' object which connects to a
    # quadstore which can be written to a file; this is prepared here
    quadstore_file = os.path.join(owl_dir, quadstore_file)
    if quadstore_file is not None:
        peo_world = owlready2.World(filename=quadstore_file)
    else:
        peo_world = owlready2.World()

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
                write_array_data=write_array_data,
            )

        # ---------------------------------------------------------------------------- #
        #             Add the problem's EXPERIMENTS to the knowledge graph             #
        # ---------------------------------------------------------------------------- #

        for exp_name, exp_dict in problem.experiments.items():

            # add the experiment as an instance; note that the notation using the
            # auxiliary variables 't1', 't2' and 't3' was chosen intentionally, since a
            # direct notation where those auxiliary variables are not used looks very
            # ugly due to the long code line which will be broken up
            t1 = iri(peo.single_experiment_data_set(exp_name))
            t2 = RDF.type
            t3 = iri(peo.single_experiment_data_set)  # type: Union[URIRef, Literal]
            graph.add((t1, t2, t3))

            # connect the experiment with its forward model; note that it is not a
            # problem that this forward model has not been added to the graph yet; the
            # connection is assigned in both directions
            t1 = iri(peo.single_experiment_data_set(exp_name))
            t2 = iri(peo.is_modelled_by)
            t3 = iri(peo.forward_model(exp_dict["forward_model"]))
            graph.add((t1, t2, t3))
            t1 = iri(peo.forward_model(exp_dict["forward_model"]))
            t2 = iri(peo.models_experiment)
            t3 = iri(peo.single_experiment_data_set(exp_name))
            graph.add((t1, t2, t3))

            # create a namespace for the considered experiment; this is required since
            # different experiments can contain data for sensors with similar names;
            # e.g. 'exp_1' and 'exp_2' can both have sensor values for a sensor 'x'
            exp_namespace_iri = new_namespace_iri(exp_name)
            graph.bind(exp_name, URIRef(exp_namespace_iri))
            namespace = peo.get_namespace(exp_namespace_iri)

            # now, add the sensor values of the experiment; this is essentially a
            # constant, so we can use the add_constant_to_graph function
            with namespace:
                for sensor_name, sensor_value in exp_dict["sensor_values"].items():
                    add_constant_to_graph(
                        peo,
                        graph,
                        sensor_value,
                        sensor_name,
                        "experiment",
                        f"Sensor value(s) from experiment '{exp_name}'",
                        include_explanations=include_explanations,
                        write_array_data=write_array_data,
                    )
                    # assign the added constant to the experiment; note that 't1' is
                    # the experiment-instance from above
                    t1 = iri(peo.single_experiment_data_set(exp_name))
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

            # add the parameter's info
            if include_explanations:
                t1 = iri(peo.parameter(prm_name))
                t2 = iri(peo.has_explanation)
                t3 = Literal(problem.parameters[prm_name].info, datatype=XSD.string)
                graph.add((t1, t2, t3))

            # -------------------- #
            #        Domain        #
            # -------------------- #

            # since a parameter is a variable, its domain is added here; since vector-
            # parameters are allowed we have to account for the parameters dimension
            for i in range(problem.parameters[prm_name].dim):
                scalar_domain = problem.parameters[prm_name].domain[i]
                # add the domain instance
                if problem.parameters[prm_name].dim == 1:
                    domain_name = f"domain_{prm_name}"
                else:
                    domain_name = f"domain_{prm_name}_index_{i}"
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = RDF.type
                t3 = iri(peo.one_dimensional_interval)
                graph.add((t1, t2, t3))
                # write the bounds of the domain
                if write_array_data:
                    b_str = [f"{domain_name}_lower_bound", f"{domain_name}_upper_bound"]
                    for jj, element_name in enumerate(b_str):
                        # add instance
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = RDF.type
                        t3 = iri(peo.scalar_constant)
                        graph.add((t1, t2, t3))
                        # add value
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = iri(peo.has_value)
                        if jj == 0:
                            t3 = Literal(scalar_domain.lower_bound, datatype=XSD.float)
                        else:
                            t3 = Literal(scalar_domain.upper_bound, datatype=XSD.float)
                        graph.add((t1, t2, t3))
                        # add column index (always zero in this case)
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = iri(peo.has_column_index)
                        t3 = Literal(0, datatype=XSD.int)
                        graph.add((t1, t2, t3))
                        # add row index
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = iri(peo.has_row_index)
                        t3 = Literal(jj, datatype=XSD.int)
                        graph.add((t1, t2, t3))
                        # associate scalar instance with vector instance and vice versa
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = URIRef(urllib.parse.unquote(part_of_iri))  # type: ignore
                        t3 = iri(peo.one_dimensional_interval(domain_name))
                        graph.add((t1, t2, t3))
                        t1 = iri(peo.one_dimensional_interval(domain_name))
                        t2 = URIRef(urllib.parse.unquote(has_part_iri))  # type: ignore
                        t3 = iri(peo.scalar_constant(element_name))
                        graph.add((t1, t2, t3))
                        # add additional information
                        if include_explanations:
                            # add the explanation
                            if jj == 0:
                                explanation = f"Lower bound of domain {domain_name}"
                            else:
                                explanation = f"Upper bound of domain {domain_name}"
                            t1 = iri(peo.scalar_constant(element_name))
                            t2 = iri(peo.has_explanation)
                            t3 = Literal(explanation, datatype=XSD.string)
                            graph.add((t1, t2, t3))
                            # add the use-string
                            t1 = iri(peo.scalar_constant(element_name))
                            t2 = iri(peo.is_used_for)
                            t3 = Literal("domain", datatype=XSD.string)
                            graph.add((t1, t2, t3))
                # add information on the inclusion of the bounds
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.lower_bound_included)
                t3 = Literal(scalar_domain.lower_bound_included, datatype=XSD.boolean)
                graph.add((t1, t2, t3))
                t1 = iri(peo.one_dimensional_interval(domain_name))
                t2 = iri(peo.upper_bound_included)
                t3 = Literal(scalar_domain.upper_bound_included, datatype=XSD.boolean)
                graph.add((t1, t2, t3))
                # add the info-string to domain instance
                if include_explanations:
                    # add the use-string to domain instance
                    t1 = iri(peo.one_dimensional_interval(domain_name))
                    t2 = iri(peo.is_used_for)
                    t3 = Literal("domain", datatype=XSD.string)
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
                t1 = iri(peo.normal_probability_density_function(prior_name))
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
                    t1 = iri(peo.normal_probability_density_function(prior_name))
                    t2 = iri(peo.has_standard_deviation)
                    if problem.parameters[std_name].is_latent:
                        t3 = iri(peo.parameter(std_name))
                    else:
                        t3 = iri(peo.constant(std_name))
                    graph.add((t1, t2, t3))
                else:
                    cov_name = f"scale_{prm_name}"
                    t1 = iri(peo.normal_probability_density_function(prior_name))
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
            elif prior_type == "uninformative":
                # uniform density function instance
                t1 = iri(peo.uniform_probability_density_function(prior_name))
                t2 = RDF.type
                t3 = iri(peo.uninformative_probability_density_function)
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

            # add the model parameters/constants as input arguments; note that it is not
            # a problem that these parameters have not been added to the graph yet
            for prm_name in fwd_model.prms_def:
                t1 = iri(peo.forward_model(fwd_name))
                if prm_name in problem.constant_prms:
                    t2 = iri(peo.has_constant)
                    t3 = iri(peo.constant(prm_name))
                else:
                    t2 = iri(peo.has_parameter)
                    t3 = iri(peo.parameter(prm_name))
                graph.add((t1, t2, t3))

            for isensor in fwd_model.input_sensors:
                with peo.get_namespace(new_namespace_iri(fwd_name)):
                    t1 = iri(peo.sensor(isensor.name))
                t2 = RDF.type
                t3 = iri(peo.sensor)
                graph.add((t1, t2, t3))
                t1 = iri(peo.forward_model(fwd_name))
                t2 = iri(peo.has_input_sensor)
                with peo.get_namespace(new_namespace_iri(fwd_name)):
                    t3 = iri(peo.sensor(isensor.name))
                graph.add((t1, t2, t3))
                # TODO: a variable needs a domain

                # input arguments from the input sensors are always specified by
                # experimental data; this connection is assigned here
                for exp_name, exp_dict in problem.experiments.items():
                    exp_namespace_iri = new_namespace_iri(exp_name)
                    with peo.get_namespace(exp_namespace_iri):
                        t1 = iri(peo.constant(isensor.name))
                    t2 = iri(peo.specifies_variable)
                    t3 = iri(peo.sensor(isensor.name))
                    graph.add((t1, t2, t3))

            for osensor in fwd_model.output_sensors:
                # create an instance of a variable (the return value)
                with peo.get_namespace(new_namespace_iri(fwd_name)):
                    t1 = iri(peo.sensor(osensor.name))
                t2 = RDF.type
                t3 = iri(peo.sensor)
                graph.add((t1, t2, t3))
                # identify the variable created before as a return value
                t1 = iri(peo.forward_model(fwd_name))
                t2 = iri(peo.has_output_sensor)
                with peo.get_namespace(new_namespace_iri(fwd_name)):
                    t3 = iri(peo.sensor(osensor.name))
                graph.add((t1, t2, t3))
                # TODO: a variable needs a domain

        # ---------------------------------------------------------------------------- #
        #          Add the problem's LIKELIHOOD MODELS to the knowledge graph          #
        # ---------------------------------------------------------------------------- #

        for likelihood_model in problem.likelihood_models.values():

            # add the likelihood model instance
            t1, t3 = identify_likelihood_model(likelihood_model, peo)
            t2 = RDF.type
            graph.add((t1, t2, t3))

            # add the likelihood model's forward model standard deviation; note that
            # this can either be a constant or a latent parameter
            t2 = iri(peo.has_forward_model_error_standard_deviation)
            const_or_parameter = get_global_name("std_model", likelihood_model.prms_def)
            if const_or_parameter in problem.constant_prms:
                t3 = iri(peo.constant(const_or_parameter))
            else:
                t3 = iri(peo.parameter(const_or_parameter))
            graph.add((t1, t2, t3))

            # add the likelihood model's optional measurement error std. deviation
            if "std_measurement" in likelihood_model.prms_def.values():
                t2 = iri(peo.has_measurement_error_standard_deviation)
                const_or_parameter = get_global_name(
                    "std_measurement", likelihood_model.prms_def
                )
                if const_or_parameter in problem.constant_prms:
                    t3 = iri(peo.constant(const_or_parameter))
                else:
                    t3 = iri(peo.parameter(const_or_parameter))
                graph.add((t1, t2, t3))

            if likelihood_model.considers_correlation:

                # add the correlation model
                cname = f"{likelihood_model.correlation_model}_correlation_model"
                t1_ = iri(peo.correlation_model(cname))
                t2 = RDF.type
                t3 = iri(peo.correlation_model)
                graph.add((t1_, t2, t3))

                # associate the correlation model with the likelihood model
                t2 = iri(peo.has_correlation_model)
                t3 = iri(peo.correlation_model(cname))
                graph.add((t1, t2, t3))

                # connect the correlation model with its parameters/constants
                for corr_var in ["l_corr", "l_corr_time", "l_corr_space"]:
                    if corr_var in likelihood_model.prms_def.values():
                        const_or_parameter = get_global_name(
                            corr_var, likelihood_model.prms_def
                        )
                        t1_ = iri(peo.correlation_model(cname))
                        t2 = iri(peo.has_input_argument)
                        if corr_var in problem.constant_prms:
                            t3 = iri(peo.constant(const_or_parameter))
                        else:
                            t3 = iri(peo.parameter(const_or_parameter))
                        graph.add((t1_, t2, t3))

            # add the experiments assigned to this likelihood model
            for exp_name in likelihood_model.experiment_names:
                t2 = iri(peo.has_experiment)
                t3 = iri(peo.single_experiment_data_set(exp_name))
                graph.add((t1, t2, t3))

            # add the likelihood's forward model
            fwd_model_name = likelihood_model.forward_model
            t2 = iri(peo.has_forward_model)
            t3 = iri(peo.forward_model(fwd_model_name))
            graph.add((t1, t2, t3))

            # add the likelihood model's sensors
            t2 = iri(peo.assesses_model_prediction_in_sensor)
            for sensor_name in likelihood_model.sensor_names:
                with peo.get_namespace(new_namespace_iri(fwd_model_name)):
                    t3 = iri(peo.variable(sensor_name))
                graph.add((t1, t2, t3))

    # print the triples to a file
    graph.serialize(destination=ttl_file, format="turtle")


def iri(s: owlready2.entity.ThingClass) -> rdflib.term.URIRef:
    """
    Gets the Internationalized Resource Identifier (IRI) from a class or an
    instance of an ontology, applies some basic parsing and returns the IRI
    as an rdflib-term as it is needed for the triple generation.
    """
    return URIRef(urllib.parse.unquote(s.iri))  # type: ignore


def add_constant_to_graph(
    peo: owlready2.namespace.Ontology,
    graph: rdflib.graph.Graph,
    constant: Union[np.ndarray, float, int],
    name: str,
    use: str,
    info: str,
    include_explanations: bool = True,
    write_array_data: bool = True,
    part_of_iri: str = "http://www.obofoundry.org/ro/#OBO_REL:part_of",
    has_part_iri: str = "http://www.obofoundry.org/ro/#OBO_REL:has_part",
):
    """
    Adds a given constant (array or scalar) to given knowledge graph.

    Parameters
    ----------
    peo
        Ontology object required to add triples in line with the parameter estimation
        ontology.
    graph
        The knowledge graph to which the given array should be added.
    constant
        The array to be added to the given graph.
    name
        The instance's name the array should be written to.
    use
        Stating what the constant is used for.
    info
        Information on what the given constant is.
    include_explanations
        If True, some of the graph's instances will have string-attributes which
        give a short explanation on what they are. If False, those explanations will
        not be included. This might be useful for graph-visualizations.
    write_array_data
        When True, the values and indices of an array are written to the graph. However,
        this might lead to a rather large graph. When False, no values and indices are
        written to the graph. This might be useful when the non-data part of the graph
        is of primary interest.
    part_of_iri
        The IRI used for the BFO object relation 'part_of'.
    has_part_iri
        The IRI used for the BFO object relation 'has_part' (inverse of part_of).
    """

    if type(constant) not in [np.ndarray, float, int]:
        raise ValueError(
            f"The given constant must be of type np.ndarray, float or int, but found "
            f"type '{type(constant)}'"
        )

    if (type(constant) is np.ndarray) and (len(constant.shape) == 1):
        # in this case the array is a flat vector, which is interpreted as a column
        # vector which means that the column index will be set to 0 for each element
        t1 = iri(peo.vector_constant(name))
        t2 = RDF.type
        t3 = iri(peo.vector_constant)  # type: Union[URIRef, Literal]
        graph.add((t1, t2, t3))
        # add the info-string if requested
        if include_explanations:
            t1 = iri(peo.vector_constant(name))
            t2 = iri(peo.has_explanation)
            t3 = Literal(info, datatype=XSD.string)
            graph.add((t1, t2, t3))
        if write_array_data:
            for row_idx, value in enumerate(constant):
                # an element of a vector is a scalar
                element_name = f"{name}_{row_idx}"
                t1 = iri(peo.scalar_constant(element_name))
                t2 = RDF.type
                t3 = iri(peo.scalar_constant)
                graph.add((t1, t2, t3))
                # add value
                t1 = iri(peo.scalar_constant(element_name))
                t2 = iri(peo.has_value)
                t3 = Literal(value, datatype=XSD.float)
                graph.add((t1, t2, t3))
                # add column index (always one in this case)
                t1 = iri(peo.scalar_constant(element_name))
                t2 = iri(peo.has_column_index)
                t3 = Literal(0, datatype=XSD.int)
                graph.add((t1, t2, t3))
                # add row index
                t1 = iri(peo.scalar_constant(element_name))
                t2 = iri(peo.has_row_index)
                t3 = Literal(row_idx, datatype=XSD.int)
                graph.add((t1, t2, t3))
                # associate scalar instance with vector instance
                t1 = iri(peo.scalar_constant(element_name))
                t2 = URIRef(urllib.parse.unquote(part_of_iri))  # type: ignore
                t3 = iri(peo.vector_constant(name))
                graph.add((t1, t2, t3))
                t1 = iri(peo.vector_constant(name))
                t2 = URIRef(urllib.parse.unquote(has_part_iri))  # type: ignore
                t3 = iri(peo.scalar_constant(element_name))
                graph.add((t1, t2, t3))
                # add additional information
                if include_explanations:
                    # add the explanation
                    explanation = f"Element of vector constant {name}"
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = iri(peo.has_explanation)
                    t3 = Literal(explanation, datatype=XSD.string)
                    graph.add((t1, t2, t3))
                    # add the use-string
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = iri(peo.is_used_for)
                    t3 = Literal(use, datatype=XSD.string)
                    graph.add((t1, t2, t3))

    elif (type(constant) is np.ndarray) and (len(constant.shape) == 2):
        # in this case we have a matrix with row and column index
        t1 = iri(peo.matrix_constant(name))
        t2 = RDF.type
        t3 = iri(peo.matrix_constant)
        graph.add((t1, t2, t3))
        if write_array_data:
            for col_idx, array_row in enumerate(constant):
                for row_idx, value in enumerate(array_row):
                    # an element of a matrix is a scalar
                    element_name = f"{name}_{row_idx}_{col_idx}"
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = RDF.type
                    t3 = iri(peo.scalar_constant)
                    graph.add((t1, t2, t3))
                    # add value
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = iri(peo.has_value)
                    t3 = Literal(value, datatype=XSD.float)
                    graph.add((t1, t2, t3))
                    # add row index
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = iri(peo.has_row_index)
                    t3 = Literal(row_idx, datatype=XSD.int)
                    graph.add((t1, t2, t3))
                    # add column index
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = iri(peo.has_column_index)
                    t3 = Literal(col_idx, datatype=XSD.int)
                    graph.add((t1, t2, t3))
                    # associate scalar instance with matrix instance
                    t1 = iri(peo.scalar_constant(element_name))
                    t2 = URIRef(urllib.parse.unquote(has_part_iri))  # type: ignore
                    t3 = iri(peo.matrix_constant(name))
                    graph.add((t1, t2, t3))
                    t1 = iri(peo.matrix_constant(name))
                    t2 = URIRef(urllib.parse.unquote(has_part_iri))  # type: ignore
                    t3 = iri(peo.scalar_constant(element_name))
                    graph.add((t1, t2, t3))
                    # add additional information
                    if include_explanations:
                        # add the explanation
                        explanation = f"Element of matrix constant {name}"
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = iri(peo.has_explanation)
                        t3 = Literal(explanation, datatype=XSD.string)
                        graph.add((t1, t2, t3))
                        # add the use-string
                        t1 = iri(peo.scalar_constant(element_name))
                        t2 = iri(peo.is_used_for)
                        t3 = Literal(use, datatype=XSD.string)
                        graph.add((t1, t2, t3))
    else:
        # in this case 'constant' represents a single number, hence a scalar
        t1 = iri(peo.scalar_constant(name))
        t2 = RDF.type
        t3 = iri(peo.scalar_constant)
        graph.add((t1, t2, t3))
        # add the scalar's value as a float
        t1 = iri(peo.scalar_constant(name))
        t2 = iri(peo.has_value)
        t3 = Literal(float(constant), datatype=XSD.float)
        graph.add((t1, t2, t3))
        # add additional information
        if include_explanations:
            # add the explanation
            t1 = iri(peo.scalar_constant(name))
            t2 = iri(peo.has_explanation)
            t3 = Literal(info, datatype=XSD.string)
            graph.add((t1, t2, t3))
            # add the use-string
            t1 = iri(peo.scalar_constant(name))
            t2 = iri(peo.is_used_for)
            t3 = Literal(use, datatype=XSD.string)
            graph.add((t1, t2, t3))


def identify_likelihood_model(likelihood_model, peo):

    if likelihood_model.considers_correlation:

        # -------------------------------- #
        #       Only time correlated       #
        # -------------------------------- #
        if likelihood_model.considers_only_time_correlation:
            if likelihood_model.additive_model_error:
                t3 = iri(peo.additive_only_time_correlated_Gaussian_likelihood_model)
                t1 = iri(
                    peo.additive_only_time_correlated_Gaussian_likelihood_model(
                        likelihood_model.name
                    )
                )
            else:
                assert likelihood_model.multiplicative_model_error
                t3 = iri(
                    peo.multiplicative_only_time_correlated_Gaussian_likelihood_model
                )
                t1 = iri(
                    peo.multiplicative_only_time_correlated_Gaussian_likelihood_model(
                        likelihood_model.name
                    )
                )

        # --------------------------------- #
        #       Only space correlated       #
        # --------------------------------- #
        elif likelihood_model.considers_only_space_correlation:
            if likelihood_model.additive_model_error:
                t3 = iri(peo.additive_only_space_correlated_Gaussian_likelihood_model)
                t1 = iri(
                    peo.additive_only_space_correlated_Gaussian_likelihood_model(
                        likelihood_model.name
                    )
                )
            else:
                assert likelihood_model.multiplicative_model_error
                t3 = iri(
                    peo.multiplicative_only_space_correlated_Gaussian_likelihood_model
                )
                t1 = iri(
                    peo.multiplicative_only_space_correlated_Gaussian_likelihood_model(
                        likelihood_model.name
                    )
                )

        # --------------------------------- #
        #       Time-space correlated       #
        # --------------------------------- #
        else:
            assert likelihood_model.considers_space_and_time_correlation
            if likelihood_model.additive_model_error:
                t3 = iri(peo.additive_time_space_correlated_Gaussian_likelihood_model)
                t1 = iri(
                    peo.additive_time_space_correlated_Gaussian_likelihood_model(
                        likelihood_model.name
                    )
                )
            else:
                assert likelihood_model.multiplicative_model_error
                t3 = iri(
                    peo.multiplicative_time_space_correlated_Gaussian_likelihood_model
                )
                t1 = iri(
                    peo.multiplicative_time_space_correlated_Gaussian_likelihood_model(
                        likelihood_model.name
                    )
                )

    # -------------------------------- #
    #           Uncorrelated           #
    # -------------------------------- #
    else:
        if likelihood_model.additive_model_error:
            t3 = iri(peo.additive_uncorrelated_Gaussian_likelihood_model)
            t1 = iri(
                peo.additive_uncorrelated_Gaussian_likelihood_model(
                    likelihood_model.name
                )
            )
        else:
            assert likelihood_model.multiplicative_model_error
            t3 = iri(peo.multiplicative_uncorrelated_Gaussian_likelihood_model)
            t1 = iri(
                peo.multiplicative_uncorrelated_Gaussian_likelihood_model(
                    likelihood_model.name
                )
            )

    return t1, t3
