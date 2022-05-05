# standard library imports
import unittest

# third party imports
from scipy import stats
import numpy as np

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.interface.knowledge_graph_export import export_knowledge_graph


class TestProblem(unittest.TestCase):
    def test_export_parameters_and_constants(self):

        problem = InverseProblem("TestProblem")
        problem.add_parameter(
            "m",
            "model",
            prior=("normal", {"mean": 0, "std": 1}),
            info="info m",
            tex="$m$",
        )
        problem.add_parameter(
            "b",
            "model",
            prior=("uniform", {"low": 2, "high": 3}),
            info="info b",
            tex="$b$",
        )
        problem.add_parameter(
            "sigma", "likelihood", domain="(0, +oo)", info="info sigma", tex=r"$\sigma$"
        )

        export_knowledge_graph(problem, "./test_graph.ttl", data_dir="./")


if __name__ == "__main__":
    unittest.main()
