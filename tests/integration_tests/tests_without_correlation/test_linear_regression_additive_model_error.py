"""
                                   Linear regression
----------------------------------------------------------------------------------------
                       ---> Additive model prediction error <---
----------------------------------------------------------------------------------------
The model equation is y(x) = a * x + b with a, b being the model parameters, while the
likelihood model is based on a normal zero-mean additive model error distribution with
the standard deviation to infer. The problem is approached via max likelihood estimation
and via sampling using emcee and dynesty.
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
from probeye.definition.distribution import Normal, TruncNormal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (knowledge graph)
from probeye.ontology.knowledge_graph_export import export_knowledge_graph

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_linear_regression_additive_model_error(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        write_to_graph: bool = True,
        run_scipy: bool = True,
        run_emcee: bool = True,
        run_dynesty: bool = True,
    ):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        n_steps
            Number of steps (samples) to run. Note that the default number is rather low
            just so the test does not take too long.
        n_initial_steps
            Number of steps for initial (burn-in) sampling.
        n_walkers
            Number of walkers used by the estimator.
        plot
            If True, the data and the posterior distributions are plotted. This is
            deactivated by default, so that the test does not stop until the generated
            plots are closed.
        show_progress
            If True, progress-bars will be shown, if available.
        write_to_graph
            Triggers the export of the solver results to a given knowledge graph.
        run_scipy
            If True, the problem is solved with scipy (maximum likelihood est).
            Otherwise, no maximum likelihood estimate is derived.
        run_emcee
            If True, the problem is solved with the emcee solver. Otherwise, the emcee
            solver will not be used.
        run_dynesty
            If True, the problem is solved with the dynesty solver. Otherwise, the
            dynesty solver will not be used.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # 'true' value of a, and its normal prior parameters
        a_true = 2.5
        mean_a = 2.0
        std_a = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        mean_b = 1.0
        std_b = 1.0

        # 'true' value of additive error sd, and its trunc. normal prior parameters
        sigma = 0.5
        low_sigma = 0.0
        high_sigma = 1.4
        mean_sigma = 0.7
        std_sigma = 1.0

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inverse problem with a useful name
        problem = InverseProblem("Linear regression (AME)")

        # add all parameters to the problem
        problem.add_parameter(
            name="a",
            tex="$a$",
            info="Slope of the graph",
            prior=Normal(mean=mean_a, std=std_a),
        )
        problem.add_parameter(
            name="b",
            info="Intersection of graph with y-axis",
            tex="$b$",
            prior=Normal(mean=mean_b, std=std_b),
        )
        problem.add_parameter(
            name="sigma",
            domain="(0, +oo)",
            tex=r"$\sigma$",
            info="Standard deviation, of zero-mean additive model error",
            prior=TruncNormal(
                low=low_sigma, high=high_sigma, mean=mean_sigma, std=std_sigma
            ),
        )

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; normal likelihood with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = a_true * x_test + b_true
        y_test = np.random.normal(loc=y_true, scale=sigma)

        # add the experimental data
        problem.add_experiment(
            name=f"TestSeries_1",
            sensor_data={
                "x": x_test,
                "y": y_test,
            },
        )

        # plot the true and noisy data
        if plot:
            plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
            plt.plot(x_test, y_true, label="true", c="black")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def interface(self):
                self.parameters = [{"a": "m"}, "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        # add the forward model to the problem
        linear_model = LinearModel("LinearModel")
        problem.add_forward_model(linear_model, experiments="TestSeries_1")

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # add the likelihood model to the problem
        problem.add_likelihood_model(
            GaussianLikelihoodModel(
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
        true_values = {"a": a_true, "b": b_true, "sigma": sigma}
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=plot,
            show_progress=show_progress,
            write_to_graph=write_to_graph,
            knowledge_graph_file=knowledge_graph_file,
            data_dir=dir_path,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_dynesty=run_dynesty,
        )

        # reduce the number of latent parameters to two; this is done to check if some
        # plotting routines also work in this setup
        problem.change_parameter_role("sigma", value=sigma)
        true_values = {"a": a_true, "b": b_true}
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=plot,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_dynesty=run_dynesty,
        )

        # again, only two parameters; this case is done to have a uniform prior plotted
        # on the vertical axis in a 2-parameter pairplot
        problem.change_parameter_role(
            "sigma", prior=Uniform(low=low_sigma, high=high_sigma)
        )
        problem.change_parameter_role("b", value=b_true)
        true_values = {"a": a_true, "sigma": sigma}
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=plot,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_dynesty=run_dynesty,
        )

        # reduce the number of latent parameters to one; this is done to check if some
        # plotting routines also work in this setup
        problem.change_parameter_role("sigma", value=sigma)
        true_values = {"a": a_true}
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=plot,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_dynesty=run_dynesty,
        )


if __name__ == "__main__":
    unittest.main()
