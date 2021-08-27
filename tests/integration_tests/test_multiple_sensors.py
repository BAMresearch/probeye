"""
Linear model in time and space with three different noise models
--------------------------------------------------------------------------------
The model equation is y = A * x + B * t with A, B being the model parameters,
while x and t represent position and time respectively. Measurements are made
at three different positions (x-values) each of which is associated with an own
zero-mean, uncorrelated normal noise model with the std. deviations to infer.
This results in five calibration parameters (parameters to infer). The problem
is solved via sampling by means of taralli.
"""

# standard library imports
import logging

# third party imports
import unittest
import numpy as np

# local imports
from probeye.definition.forward_model import ModelTemplate
from probeye.definition.forward_model import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.noise_model import NormalNoiseZeroMean
from probeye.inference.taralli_.solver import taralli_solver


class TestProblem(unittest.TestCase):

    def test_multiple_sensors(self, n_steps=100, n_walkers=20, plot=False,
                              verbose=False):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps (samples) to run. Note that the default number is
            rather low just so the test does not take too long.
        n_walkers : int, optional
            Number of walkers used by the estimator.
        plot : bool, optional
            If True, the data and the posterior distributions are plotted. This
            is deactivated by default, so that the test does not stop until the
            generated plots are closed.
        verbose : bool, optional
            If True, additional information will be printed to the console.
        """

        # ==================================================================== #
        #                          Set numeric values                          #
        # ==================================================================== #

        # 'true' value of A, and its normal prior parameters
        A_true = 42.0
        loc_A = 40.0
        scale_A = 5.0

        # 'true' value of B, and its normal prior parameters
        B_true = 6174.0
        loc_B = 6000.0
        scale_B = 300.0

        # 'true' value of sd_S1, and its uniform prior parameters
        sd_S1_true = 0.2
        low_S1 = 0.1
        high_S1 = 0.7

        # 'true' value of sd_S2, and its uniform prior parameters
        sd_S2_true = 0.4
        low_S2 = 0.1
        high_S2 = 0.7

        # 'true' value of sd_S3, and its uniform prior parameters
        sd_S3_true = 0.6
        low_S3 = 0.1
        high_S3 = 0.7

        # define sensor positions
        pos_s1 = 0.2
        pos_s2 = 0.5
        pos_s3 = 42.0

        # ==================================================================== #
        #                       Define the Forward Model                       #
        # ==================================================================== #

        class PositionSensor(Sensor):
            def __init__(self, name, position):
                super().__init__(name)
                self.position = position

        class LinearModel(ModelTemplate):
            def __call__(self, inp):
                t = inp['time']
                A = inp['A']
                B = inp['B']
                response_dict = dict()
                for os in self.output_sensors:
                    response_dict[os.name] = A * os.position + B * t
                return response_dict

        # ==================================================================== #
        #                     Define the Inference Problem                     #
        # ==================================================================== #

        # initialize the inference problem with a useful name
        problem = InferenceProblem("Linear model with three noise models")

        # add all parameters to the problem
        problem.add_parameter('A', 'model',
                              prior=('normal', {'loc': loc_A,
                                                'scale': scale_A}),
                              info="Slope of the graph",
                              tex="$A$")
        problem.add_parameter('B', 'model',
                              prior=('normal', {'loc': loc_B,
                                                'scale': scale_B}),
                              info="Intersection of graph with y-axis",
                              tex='$B$')
        problem.add_parameter('sigma_1', 'noise',
                              prior=('uniform', {'low': low_S1,
                                                 'high': high_S1}),
                              info="Std. dev. of zero-mean noise model for S1",
                              tex=r"$\sigma_1$")
        problem.add_parameter('sigma_2', 'noise',
                              prior=('uniform', {'low': low_S2,
                                                 'high': high_S2}),
                              info="Std. dev. of zero-mean noise model for S1",
                              tex=r"$\sigma_2$")
        problem.add_parameter('sigma_3', 'noise',
                              prior=('uniform', {'low': low_S3,
                                                 'high': high_S3}),
                              info="Std. dev. of zero-mean noise model for S1",
                              tex=r"$\sigma_3$")

        # add the forward model to the problem
        inp_1 = Sensor("time")
        out_1 = PositionSensor("S1", pos_s1)
        out_2 = PositionSensor("S2", pos_s2)
        out_3 = PositionSensor("S3", pos_s3)
        linear_model = LinearModel(['A', 'B'], [inp_1], [out_1, out_2, out_3])
        problem.add_forward_model("LinearModel", linear_model)

        # add the noise model to the problem
        problem.add_noise_model(out_1.name, NormalNoiseZeroMean(['sigma_1']))
        problem.add_noise_model(out_2.name, NormalNoiseZeroMean(['sigma_2']))
        problem.add_noise_model(out_3.name, NormalNoiseZeroMean(['sigma_3']))

        # ==================================================================== #
        #                Add test data to the Inference Problem                #
        # ==================================================================== #

        # add the experimental data
        np.random.seed(1)
        sd_dict = {out_1.name: sd_S1_true,
                   out_2.name: sd_S2_true,
                   out_3.name: sd_S3_true}

        def generate_data(n_time_steps, n=None):
            time_steps = np.linspace(0, 1, n_time_steps)
            inp = {'A': A_true, 'B': B_true, 'time': time_steps}
            sensors = linear_model(inp)
            for key, val in sensors.items():
                sensors[key] = val + np.random.normal(0.0, sd_dict[key],
                                                      size=n_time_steps)
            sensors['time'] = time_steps
            problem.add_experiment(f'TestSeries_{n}',
                                   sensor_values=sensors,
                                   fwd_model_name='LinearModel')
        for n_exp, n_t in enumerate([101, 51]):
            generate_data(n_t, n=n_exp)

        # give problem overview
        if verbose:
            problem.info()

        # ==================================================================== #
        #                      Solve problem with Taralli                      #
        # ==================================================================== #

        # run the taralli solver with deactivated output
        logging.root.disabled = True
        taralli_solver(problem, n_walkers=n_walkers, n_steps=n_steps,
                       plot=plot, summary=verbose)

if __name__ == "__main__":
    unittest.main()
