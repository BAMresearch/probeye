# standard library
import os
from typing import Union, Optional
import urllib

# third party imports
import numpy as np
import owlready2
import rdflib
from rdflib import URIRef, Graph, Literal

# noinspection PyProtectedMember
from rdflib.namespace import RDF, XSD

# local imports
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.subroutines import get_global_name


def export_knowledge_graph(
    problem: InferenceProblem,
    ttl_file: str,
    owl_basename: str = "parameter_estimation_ontology.owl",
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