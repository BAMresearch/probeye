"""
                        Linear regression (max. likelihood only)
----------------------------------------------------------------------------------------
                       ---> Additive model prediction error <---
----------------------------------------------------------------------------------------
The model equation is y(x) = a * x + b with a, b being the model parameters, while the
likelihood model is based on a normal zero-mean additive model error distribution with
the standard deviation to infer. The problem is approached via max likelihood. This
example is intended to show the minimal problem setup if only a maximum likelihood fit
is intended (in this case, no priors are needed to define latent parameters).
"""

# standard library imports
import unittest
import os

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (knowledge graph)
from probeye.ontology.knowledge_graph_export import export_knowledge_graph
from probeye.ontology.knowledge_graph_export import export_results_to_knowledge_graph

# local imports (inference related)
from probeye.inference.scipy.solver import ScipySolver


class TestProblem(unittest.TestCase):
    def test_maximum_likelihood(self, plot: bool = False, write_to_graph: bool = True):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        plot
            If True, the problem's data is plotted. This is deactivated by default, so
            that the test does not stop until the generated plots are closed.
        write_to_graph
            Triggers the export of the solver results to a given knowledge graph.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # 'true' value of m
        m_true = 2.5

        # 'true' value of b
        b_true = 1.7

        # 'true' value of additive model error sd
        sigma = 0.5

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["m", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp: dict) -> dict:
                return {"y": inp["m"] * inp["x"] + inp["b"]}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inverse problem with a useful name
        problem = InverseProblem("Linear regression (AME)")

        # add all parameters to the problem
        problem.add_parameter("m", "model")
        problem.add_parameter("b", "model")
        problem.add_parameter("sigma", "likelihood", domain="(0, +oo)")

        # add the forward model to the problem
        linear_model = LinearModel("LinearModel")
        problem.add_forward_model(linear_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; normal likelihood with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = linear_model.response(
            {linear_model.input_sensor.name: x_test, "m": m_true, "b": b_true}
        )[linear_model.output_sensor.name]
        y_test = np.random.normal(loc=y_true, scale=sigma)

        # add the experimental data
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={
                linear_model.input_sensor.name: x_test,
                linear_model.output_sensor.name: y_test,
            },
        )

        # plot the true and noisy data
        if plot:
            plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
            plt.plot(x_test, y_true, label="true", c="black")
            plt.xlabel(linear_model.input_sensor.name)
            plt.ylabel(linear_model.output_sensor.name)
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # add the likelihood model to the problem
        problem.add_likelihood_model(
            GaussianLikelihoodModel(
                prms_def="sigma",
                experiment_name="TestSeries_1",
                model_error="additive",
            )
        )

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                            Export knowledge graph                            #
        # ============================================================================ #

        # create the knowledge graph and print it to file
        dir_path = os.path.dirname(__file__)
        basename_owl = os.path.basename(__file__).split(".")[0] + ".owl"
        knowledge_graph_file = os.path.join(dir_path, basename_owl)
        export_knowledge_graph(problem, knowledge_graph_file, data_dir=dir_path)

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way; ref_values are used for plotting
        true_values = {"m": m_true, "b": b_true, "sigma": sigma}
        scipy_solver = ScipySolver(problem)
        inference_data = scipy_solver.run_max_likelihood(true_values=true_values)
        if write_to_graph:
            export_results_to_knowledge_graph(
                problem,
                inference_data,
                knowledge_graph_file,
                data_dir=dir_path,
            )


if __name__ == "__main__":
    unittest.main()
