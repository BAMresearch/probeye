# third party imports
import numpy as np
from owlready2 import default_world, get_ontology


def import_parameter_samples(knowledge_graph_file: str) -> dict:
    """
    Reads in a knowledge graph file (an owl-file) which contains the results of some
    sampling-based solver, and returns the parameter samples as a dictionary.

    Parameters
    ----------
    knowledge_graph_file
        The owl-file that contains an inverse problem and the respective joint posterior
        distribution.

    Returns
    -------
    sample_dict
        The keys are the parameter names of the given problem, and the values are the
        numpy arrays (flat vectors) of the respective samples.
    """

    # load the given knowledge graph from file
    get_ontology(knowledge_graph_file).load()

    # query the graph; the result will be a list of lists with three entries (like in
    # [[e1, e2, e3], [e1, e2, e3]]) where the first elements contain the parameter names
    # the second elements are the row indices in the posterior sample file, and the last
    # elements should be all equal stating the file of the joint posterior samples
    query_result_raw = list(
        default_world.sparql(
            """
            PREFIX peo: <http://www.parameter_estimation_ontology.org#>
            SELECT ?x ?index ?f
            WHERE { ?i a peo:inverse_problem .
                    ?i peo:has_joint_posterior_distribution ?d .
                    ?d peo:has_samples ?s .
                    ?s peo:has_file ?f .
                    ?i peo:has_parameter ?x .
                    ?x peo:has_posterior_index ?index}
            """
        )
    )

    # get the filename of the joint posterior's samples
    filenames_list = [e[2] for e in query_result_raw]
    filenames_set = set(filenames_list)
    assert len(filenames_set) == 1
    filename = filenames_list[0]

    # load the samples from the file
    samples_array = np.loadtxt(filename)

    # post-process the raw query results into a dictionary
    sample_dict = {}
    for sublist in query_result_raw:
        prm_name = str(sublist[0]).split(".")[-1]
        idx = sublist[1]
        sample_dict[prm_name] = samples_array[idx]

    return sample_dict
