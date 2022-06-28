"""
                         Linear regression with 1D correlation
----------------------------------------------------------------------------------------
                       ---> Additive model prediction error <---
----------------------------------------------------------------------------------------
The n data points (y1, y2, ..., yn) generated for this example are sampled from an
n-variate normal distribution with mean values given by yi = a * xi + b with a, b being
the model parameters and x1, x2, ..., xi, ..., xn being predefined spatial x-coordinates
ranging from 0 to 1. The data points (y1, y2, ..., yn) are not independent but
correlated in x. This means, the closer yi and yj are in terms of x (i.e., the smaller
|xi - xj|) the greater the correlation between yi and yj. The corresponding covariance
matrix is defined based on an exponential correlation function parameterized by the
constant standard deviation sigma of the n-variate normal distribution and a correlation
length l_corr. Hence, the full model has four parameters a, b, sigma, l_corr, all of
which are inferred in this example using a maximum likelihood estimation. Note that this
test just provides an alternative definition of the correlation to the integration test
test_1D_correlation_additive_model_error.py.
"""

# standard library
import unittest
import os

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from tripy.utils import correlation_function
from tripy.utils import correlation_matrix

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.definition.correlation_model import ExpModel

# local imports (knowledge graph)
from probeye.ontology.knowledge_graph_export import export_knowledge_graph

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_space_1D_correlation_additive_model_error(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        write_to_graph: bool = True,
        run_scipy: bool = True,
        run_emcee: bool = False,  # intentionally False for faster test-runs
        run_dynesty: bool = False,  # intentionally False for faster test-runs
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

        # 'true' value of additive error sd, and its uniform prior parameters
        sigma = 0.1
        low_sigma = 0.0
        high_sigma = 0.5

        # 'true' value of correlation length, and its uniform prior parameters
        l_corr = 0.05
        low_l_corr = 0.0
        high_l_corr = 0.2

        # settings for the data generation
        n_experiments = 3
        n_points = 25
        seed = 1

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inverse problem with a useful name
        problem = InverseProblem("Linear regression with 1D correlation (AME)")

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
            info="Standard deviation of zero-mean additive model error",
            prior=Uniform(low=low_sigma, high=high_sigma),
        )
        problem.add_parameter(
            name="l_corr",
            domain="(0, +oo)",
            tex=r"$l_\mathrm{corr}$",
            info="Correlation length of correlation model",
            prior=Uniform(low=low_l_corr, high=high_l_corr),
        )

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; first create the true values without an error model; these
        # 'true' values will be the mean values for sampling from a multivariate normal
        # distribution that accounts for the intended correlation
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_points)
        y_true = a_true * x_test + b_true

        # assemble the spatial covariance matrix
        x_test_as_column_matrix = x_test.reshape((n_points, -1))
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov = sigma**2 * correlation_matrix(x_test_as_column_matrix, f_corr)

        # now generate the noisy test data including correlations; we assume here that
        # there are n_experiments test series
        for i in range(n_experiments):
            y_test = np.random.multivariate_normal(mean=y_true, cov=cov)
            problem.add_experiment(
                name=f"Test_{i}",
                sensor_data={f"y{i}": float(y_test[i]) for i in range(n_points)},
            )
            if plot:
                plt.scatter(
                    x_test,
                    y_test,
                    label=f"measured data (test {i + 1})",
                    s=10,
                    zorder=10,
                )
        # finish the plot
        if plot:
            plt.plot(x_test, y_true, label="true model", c="black", linewidth=3)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["a", "b"]
                self.input_sensors = []
                self.output_sensors = [
                    Sensor(f"y{ii}", x=float(x_test[ii]), std_model="sigma")
                    for ii in range(n_points)
                ]

            def response(self, inp: dict) -> dict:
                a = inp["a"]
                b = inp["b"]
                return {
                    f"y{j}": a * osensor.x + b
                    for j, osensor in enumerate(self.output_sensors)
                }

        # add the forward model to the problem
        linear_model = LinearModel("LinearModel")
        problem.add_forward_model(
            linear_model, experiments=[f"Test_{i}" for i in range(n_experiments)]
        )

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # each likelihood model is assigned exactly one experiment
        for i in range(n_experiments):
            likelihood_model = GaussianLikelihoodModel(
                experiment_name=f"Test_{i}",
                model_error="additive",
                correlation=ExpModel(x="l_corr"),
            )
            problem.add_likelihood_model(likelihood_model)

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
        # integration tests in the same way
        true_values = {"a": a_true, "b": b_true, "sigma": sigma, "l_corr": l_corr}
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


if __name__ == "__main__":
    unittest.main()
