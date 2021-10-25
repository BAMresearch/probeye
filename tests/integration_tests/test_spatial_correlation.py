"""
Linear regression example with spatial correlation model
--------------------------------------------------------------------------------
The n data points (y1, y2, ..., yn) generated for this example are sampled from
an n-variate normal distribution with mean values given by yi = a * xi + b with
a, b being the model parameters and x1, x2, ..., xi, ..., xn being predefined
spatial x-coordinates ranging from 0 to 1. The data points (y1, y2, ..., yn) are
not independent but correlated. The corresponding covariance matrix is defined
based on an exponential correlation function parameterized by the const standard
deviation sigma of the n-variate normal distribution and a correlation length
l_corr. Hence, the full model has four parameters a, b, sigma, l_corr, all of
which are inferred in this example using emcee-sampling.
"""

# standard library
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.noise_model import NormalNoiseModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines
from probeye.inference.scipy_.correlation_models import \
    SpatialExponentialCorrelationModel


class TestProblem(unittest.TestCase):

    def test_spatial_correlation(self, n_steps=200, n_initial_steps=100,
                                 n_walkers=20, plot=False, verbose=False,
                                 run_scipy=True, run_emcee=True,
                                 run_torch=False):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps (samples) to run. Note that the default number is
            rather low just so the test does not take too long.
        n_initial_steps : int, optional
            Number of steps for initial (burn-in) sampling.
        n_walkers : int, optional
            Number of walkers used by the estimator.
        plot : bool, optional
            If True, the data and the posterior distributions are plotted. This
            is deactivated by default, so that the test does not stop until the
            generated plots are closed.
        verbose : bool, optional
            If True, additional information will be printed to the console.
        run_scipy : bool, optional
            If True, the problem is solved with scipy (maximum likelihood est).
            Otherwise, no maximum likelihood estimate is derived.
        run_emcee : bool, optional
            If True, the problem is solved with the emcee solver. Otherwise,
            the emcee solver will not be used.
        run_torch : bool, optional
            If True, the problem is solved with the pyro/torch_ solver.
            Otherwise, the pyro/torch_ solver will not be used.
        """

        if run_torch:
            raise RuntimeError("The pyro-solver is not available yet for "
                               "forward models including correlations.")

        # ==================================================================== #
        #                          Set numeric values                          #
        # ==================================================================== #

        # 'true' value of a, and its normal prior parameters
        a_true = 2.5
        loc_a = 2.0
        scale_a = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        loc_b = 1.0
        scale_b = 1.0

        # 'true' value of noise sd, and its uniform prior parameters
        sigma = 0.5
        low_sigma = 0.1
        high_sigma = 0.8

        # 'true' value of correlation length, and its uniform prior parameters
        l_corr = 0.05
        low_l_corr = 0.001
        high_l_corr = 0.2

        # settings for the data generation
        n_experiments = 8
        n_points = 50
        seed = 1

        # ==================================================================== #
        #                       Define the Forward Model                       #
        # ==================================================================== #

        class LinearModel(ForwardModelBase):
            def response(self, inp):
                a = inp['a']
                b = inp['b']
                response = {}
                for os in self.output_sensors:
                    response[os.name] = a * os.x.flatten() + b
                return response

        # ==================================================================== #
        #                     Define the Inference Problem                     #
        # ==================================================================== #

        # initialize the inference problem with a useful name; note that the
        # name will only be stored as an attribute of the InferenceProblem and
        # is not important for the problem itself; can be useful when dealing
        # with multiple problems
        problem = InferenceProblem("Linear regression with normal noise")

        # add all parameters to the problem; the first argument states the
        # parameter's global name (here: 'a', 'b' and 'sigma'); the second
        # argument defines the parameter type (three options: 'model' for
        # parameter's of the forward model, 'prior' for prior parameters and
        # 'noise' for parameters of the noise model); the 'info'-argument is a
        # short description string used for logging, and the tex-argument gives
        # a tex-string of the parameter used for plotting; finally, the prior-
        # argument specifies the parameter's prior; note that this definition
        # of a prior will result in the initialization of constant parameters of
        # type 'prior' in the background
        problem.add_parameter('a', 'model',
                              tex="$a$",
                              info="Slope of the graph",
                              prior=('normal', {'loc': loc_a,
                                                'scale': scale_a}))
        problem.add_parameter('b', 'model',
                              info="Intersection of graph with y-axis",
                              tex='$b$',
                              prior=('normal', {'loc': loc_b,
                                                'scale': scale_b}))
        problem.add_parameter('sigma', 'noise',
                              tex=r"$\sigma$",
                              info="Std. dev, of 0-mean noise model",
                              prior=('uniform', {'low': low_sigma,
                                                 'high': high_sigma}))
        problem.add_parameter('l_corr', 'noise',
                              tex=r"$l_\mathrm{corr}$",
                              info="Correlation length of correlation model",
                              prior=('uniform', {'low': low_l_corr,
                                                 'high': high_l_corr}))

        # add the forward model to the problem; note that the first positional
        # argument [{'a': 'm'}, 'b'] passed to LinearModel defines the forward
        # model's parameters by name via a list with elements structured like
        # {<global parameter name>: <local parameter name>}; a global name is a
        # name introduced by problem.add_parameter, while a local name is a name
        # used in the response-method of the forward model class (see the class
        # LinearModel above); note that the use of the local parameter name 'm'
        # for the global parameter 'a' is added here only to highlight the
        # possibility of this feature; it is not necessary at all here; whenever
        # forward model's parameter has a similar local and global name (which
        # should be the case most of the times), one doesn't have to use the
        # verbose notation  {<global parameter name>: <local parameter name>}
        # but can instead just write the parameter's (global=local) name, like
        # it is done with the forward model's parameter 'b' below
        x_test = np.linspace(0.0, 1.0, n_points)
        osensor = Sensor("y", x=x_test)
        linear_model = LinearModel(['a', 'b'], [], [osensor])
        problem.add_forward_model("LinearModel", linear_model)

        # add the noise model to the problem
        noise_model = NormalNoiseModel(
            sensors=osensor, corr='x', corr_model='exp',
            prms_def=[{'sigma': 'std'}, 'l_corr'])
        problem.add_noise_model(noise_model)

        # ==================================================================== #
        #                Add test data to the Inference Problem                #
        # ==================================================================== #

        # data-generation; first create the true values without noise; these
        # true values will be the mean values for sampling from a multivariate
        # normal distribution
        np.random.seed(seed)
        y_true = linear_model({'a': a_true, 'b': b_true})[osensor.name]

        # create the covariance matrix
        correlation_model = SpatialExponentialCorrelationModel(x=osensor.x)
        cov = correlation_model({'std': sigma, 'l_corr': l_corr})

        # now generate the noisy test data including correlations; we assume
        # here that there are n_experiments test series
        for i in range(n_experiments):
            y_test = np.random.multivariate_normal(mean=y_true, cov=cov)
            problem.add_experiment(f'Test_{i}', fwd_model_name="LinearModel",
                                   sensor_values={osensor.name: y_test})
            if plot:
                plt.scatter(x_test, y_test, label=f'measured data (test {i+1})',
                            s=10, zorder=10)
        # finish the plot
        if plot:
            plt.plot(x_test, y_true, label='true model', c="black", linewidth=3)
            plt.xlabel('x')
            plt.ylabel(osensor.name)
            plt.legend()
            plt.tight_layout()
            plt.draw()  # plt.draw() does not stop execution

        # ==================================================================== #
        #                Solve problem with inference engine(s)                #
        # ==================================================================== #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way
        true_values = {'a': a_true, 'b': b_true, 'sigma': sigma,
                       'l_corr': l_corr}
        run_inference_engines(problem, true_values=true_values, n_steps=n_steps,
                              n_initial_steps=n_initial_steps,
                              n_walkers=n_walkers, plot=plot, verbose=verbose,
                              run_scipy=run_scipy, run_emcee=run_emcee,
                              run_torch=run_torch)

if __name__ == "__main__":
    unittest.main()
