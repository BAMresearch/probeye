"""
                     Linear regression (with knowledge graph query)
----------------------------------------------------------------------------------------
                       ---> Additive model prediction error <---
----------------------------------------------------------------------------------------
The model equation is y(x) = a * x + b with a, b being the model parameters, while the
likelihood model is based on a normal zero-mean additive model error distribution with
the standard deviation to infer. The problem is approached via emcee. After the problem
and the sampling results have been exported to a knowledge graph, this graph is queried
to obtain the problem's parameters and the paths to the files that contain the samples.
"""

# standard library imports
import unittest
import os

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from owlready2 import default_world, get_ontology

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (knowledge graph)
from probeye.ontology.knowledge_graph_export import export_knowledge_graph
from probeye.ontology.knowledge_graph_export import export_results_to_knowledge_graph

# local imports (testing related)
from probeye.inference.emcee.solver import EmceeSolver


class TestProblem(unittest.TestCase):
    def test_query_results(
        self,
        n_steps: int = 100,
        n_initial_steps: int = 50,
        n_walkers: int = 10,
        plot: bool = False,
        show_progress: bool = False,
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
        sigma = 0.5
        low_sigma = 0.0
        high_sigma = 0.8

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

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

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inverse problem with a useful name
        problem = InverseProblem("Linear regression (AME)")

        # add all parameters to the problem
        problem.add_parameter(
            "a",
            "model",
            tex="$a$",
            info="Slope of the graph",
            prior=Normal(mean=mean_a, std=std_a),
        )
        problem.add_parameter(
            "b",
            "model",
            info="Intersection of graph with y-axis",
            tex="$b$",
            prior=Normal(mean=mean_b, std=std_b),
        )
        problem.add_parameter(
            "sigma",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$\sigma$",
            info="Standard deviation of zero-mean additive model error",
            prior=Uniform(low=low_sigma, high=high_sigma),
        )

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
            {linear_model.input_sensor.name: x_test, "m": a_true, "b": b_true}
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
                name="SimpleLikelihoodModel",
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
        #        Solve problem with inference engine and write results to graph        #
        # ============================================================================ #

        # run inference step using emcee
        true_values = {"a": a_true, "b": b_true, "sigma": sigma}
        emcee_solver = EmceeSolver(problem, show_progress=show_progress)
        inference_data = emcee_solver.run_mcmc(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            true_values=true_values,
        )

        # export the results from the 'inference_data' object to the graph
        export_results_to_knowledge_graph(
            problem,
            inference_data,
            knowledge_graph_file,
            data_dir=dir_path,
        )

        # ============================================================================ #
        #                          Query the knowledge graph                           #
        # ============================================================================ #

        # load the given knowledge graph from file
        get_ontology(knowledge_graph_file).load()

        # query the graph; the result will be a list of lists with three entries (like
        # in [[e1, e2, e3], [e1, e2, e3]]) where the first elements contain the
        # parameter names the second elements are the row indices in the posterior
        # sample file, and the last elements should be all equal stating the file of
        # the joint posterior samples
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

        # check if the parameters returned from the query are the same as the ones
        # defined within the problem scope
        prm_names_from_query = set(sample_dict.keys())
        prm_names_in_problem = set(problem.latent_prms)
        self.assertTrue(prm_names_from_query == prm_names_in_problem)


if __name__ == "__main__":
    unittest.main()
