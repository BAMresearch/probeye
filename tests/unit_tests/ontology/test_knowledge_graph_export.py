# standard library imports
import unittest
import os

# third party imports
from scipy import stats
import numpy as np

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.distribution import TruncNormal, LogNormal, SampleBased
from probeye.ontology.knowledge_graph_export import export_knowledge_graph


class TestProblem(unittest.TestCase):
    def test_unusual_priors(self):

        problem = InverseProblem("Unusual priors")

        # add a parameter with a truncated normal prior
        problem.add_parameter("a", prior=TruncNormal(mean=0, std=1, low=-5, high=5))

        # add a parameter with a log-normal prior
        problem.add_parameter("b", prior=LogNormal(mean=0, std=1))

        # add a latent parameter with a prior that is based on samples (for example,
        # from a previous inference)
        samples = np.array([1, 2, 3, 4, 5])
        problem.add_parameter("c", prior=SampleBased(samples=samples))

        # create the knowledge graph and print it to file
        dir_path = os.path.dirname(__file__)
        basename = os.path.basename(__file__).split(".")[0] + ".owl"
        knowledge_graph_file = os.path.join(dir_path, basename)
        export_knowledge_graph(problem, knowledge_graph_file, data_dir=dir_path)


if __name__ == "__main__":
    unittest.main()
