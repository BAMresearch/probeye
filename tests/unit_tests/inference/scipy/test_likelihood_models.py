# standard library imports
import unittest

# third party imports
import numpy as np
from tripy.loglikelihood import chol_loglike_1D
from tripy.loglikelihood import kron_loglike_2D_tridiag
from tripy.loglikelihood import chol_loglike_2D
from tripy.loglikelihood import kron_loglike_2D
from tripy.loglikelihood import _loglike_multivariate_normal
from tripy.utils import correlation_matrix
from tripy.utils import correlation_function
from tripy.utils import inv_cov_vec_1D

# local imports
from probeye.definition.sensor import Sensor
from probeye.inference.scipy.likelihood_models import ScipyLikelihoodBase
from probeye.inference.scipy.likelihood_models import (
    AdditiveUncorrelatedModelError,
    AdditiveCorrelatedModelError1D,
    AdditiveSpaceCorrelatedModelError2D3D,
    AdditiveSpaceTimeCorrelatedModelError1D,
    AdditiveSpaceTimeCorrelatedModelError2D3D,
    MultiplicativeUncorrelatedModelError,
    MultiplicativeCorrelatedModelError1D,
    MultiplicativeSpaceCorrelatedModelError2D3D,
    MultiplicativeSpaceTimeCorrelatedModelError1D,
    MultiplicativeSpaceTimeCorrelatedModelError2D3D,
)


class TestProblem(unittest.TestCase):
    def test_ScipyLikelihoodBase(self):
        # check the base class initialization and loglike-method
        scipy_likelihood_base = ScipyLikelihoodBase(
            "a", Sensor("y"), "Exp_1", {}, False, "x", "exp", "L1"
        )
        with self.assertRaises(NotImplementedError):
            scipy_likelihood_base.loglike({}, {})

    def test_AdditiveUncorrelatedModelError(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {"y": dummy_data}
        problem_experiments = {
            "Exp_1": {"forward_model": "FwdModel", "sensor_values": sensor_values},
            "Exp_2": {"forward_model": "FwdModel", "sensor_values": sensor_values},
        }
        n_data_points = len(problem_experiments) * n_data_points_exp

        # checks for additive_measurement_error=False
        like_model = AdditiveUncorrelatedModelError(
            prms_def=["std_model"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        dummy_response = dummy_data
        expected_ll = -n_data_points / 2 * np.log(2 * np.pi * std_model ** 2)
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(model_response_dict, {"std_model": std_model})
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        dummy_response = dummy_data
        expected_ll = -np.infty
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(model_response_dict, {"std_model": std_model})
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = AdditiveUncorrelatedModelError(
            prms_def=["std_model"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        dummy_response = dummy_data
        std = np.sqrt(std_model ** 2 + std_measurement ** 2)
        expected_ll = -n_data_points / 2 * np.log(2 * np.pi * std ** 2)
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {"std_model": std_model, "std_measurement": std_measurement},
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        dummy_response = dummy_data
        expected_ll = -np.infty
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {"std_model": std_model, "std_measurement": std_measurement},
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_MultiplicativeUncorrelatedModelError(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {"y": dummy_data}
        problem_experiments = {
            "Exp_1": {"forward_model": "FwdModel", "sensor_values": sensor_values},
            "Exp_2": {"forward_model": "FwdModel", "sensor_values": sensor_values},
        }
        n_data_points = len(problem_experiments) * n_data_points_exp

        # checks for additive_measurement_error=False
        like_model = MultiplicativeUncorrelatedModelError(
            prms_def=["std_model"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        dummy_response = dummy_data
        dummy_response_total = np.concatenate(
            [dummy_data for _ in range(len(problem_experiments))]
        )
        expected_ll = -0.5 * (
            n_data_points * np.log(2.0 * np.pi)
            + np.sum(np.log(np.power(dummy_response_total * std_model, 2)))
        )
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(model_response_dict, {"std_model": std_model})
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        dummy_response = dummy_data
        expected_ll = -np.infty
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(model_response_dict, {"std_model": std_model})
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = MultiplicativeUncorrelatedModelError(
            prms_def=["std_model"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        dummy_response = dummy_data
        dummy_response_total = np.concatenate(
            [dummy_data for _ in range(len(problem_experiments))]
        )
        expected_ll = -0.5 * (
            n_data_points * np.log(2.0 * np.pi)
            + np.sum(
                np.log(
                    np.power(dummy_response_total * std_model, 2) + std_measurement ** 2
                )
            )
        )
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {"std_model": std_model, "std_measurement": std_measurement},
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        dummy_response = dummy_data
        expected_ll = -np.infty
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {"std_model": std_model, "std_measurement": std_measurement},
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_MultiplicativeCorrelatedModelError1D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {"x": dummy_data, "y": dummy_data}
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"y": {"x": "x"}},
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"y": {"x": "x"}},
            },
        }

        # checks for additive_measurement_error=False
        like_model = MultiplicativeCorrelatedModelError1D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="x",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict, {"std_model": std_model, "l_corr": l_corr}
        )
        expected_ll = chol_loglike_1D(
            np.zeros(n_data_points_exp),
            dummy_data,
            l_corr,
            std_model,
            None,
            y_model=dummy_data,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict, {"std_model": std_model, "l_corr": l_corr}
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr is handled correctly
        std_model = 2.0
        l_corr = -1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict, {"std_model": std_model, "l_corr": l_corr}
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = MultiplicativeCorrelatedModelError1D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="x",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        expected_ll = chol_loglike_1D(
            np.zeros(n_data_points_exp),
            dummy_data,
            l_corr,
            std_model,
            std_measurement,
            y_model=dummy_data,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_AdditiveCorrelatedModelError1D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {"x": dummy_data, "y": dummy_data}
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"y": {"x": "x"}},
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"y": {"x": "x"}},
            },
        }

        # test for the correlation variable check; here the correlation_variable 'z' is
        # not found in the experiment's correlation_info, hence an error is raised
        with self.assertRaises(RuntimeError):
            AdditiveCorrelatedModelError1D(
                prms_def=["std_model", "l_corr"],
                sensors=[Sensor("y")],
                experiment_names=["Exp_1", "Exp_2"],
                problem_experiments=problem_experiments,
                additive_measurement_error=False,
                correlation_variables="z",
                correlation_model="exp",
                name="L1",
            )

        # test for the error that is raised when all given correlation variables are
        # scalars and not vectors
        problem_experiments_error = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": {"x": 1.0, "y": dummy_data},
                "correlation_info": {"y": {"x": "x"}},
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": {"x": 2.0, "y": dummy_data},
                "correlation_info": {"y": {"x": "x"}},
            },
        }
        with self.assertRaises(RuntimeError):
            AdditiveCorrelatedModelError1D(
                prms_def=["std_model", "l_corr"],
                sensors=[Sensor("y")],
                experiment_names=["Exp_1", "Exp_2"],
                problem_experiments=problem_experiments_error,
                additive_measurement_error=False,
                correlation_variables="x",
                correlation_model="exp",
                name="L1",
            )

        # checks for additive_measurement_error=False
        like_model = AdditiveCorrelatedModelError1D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="x",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict, {"std_model": std_model, "l_corr": l_corr}
        )
        expected_ll = chol_loglike_1D(
            np.zeros(n_data_points_exp),
            dummy_data,
            l_corr,
            std_model,
            None,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict, {"std_model": std_model, "l_corr": l_corr}
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr is handled correctly
        std_model = 2.0
        l_corr = -1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict, {"std_model": std_model, "l_corr": l_corr}
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = AdditiveCorrelatedModelError1D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("y")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="x",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        expected_ll = chol_loglike_1D(
            np.zeros(n_data_points_exp),
            dummy_data,
            l_corr,
            std_model,
            std_measurement,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"y": dummy_response},
            "Exp_2": {"y": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_AdditiveSpaceCorrelatedModelError2D3D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {"x": dummy_data, "y": dummy_data, "z": dummy_data}
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"z": {"x": "x", "y": "y"}},
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"z": {"x": "x", "y": "y"}},
            },
        }

        # checks for additive_measurement_error=False
        like_model = AdditiveSpaceCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("z")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="xy",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr": l_corr,
            },
        )
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        space_vector = np.zeros((n_data_points_exp, 2))
        space_vector[:, 0] = dummy_data
        space_vector[:, 1] = dummy_data
        cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f_corr)
        expected_ll = _loglike_multivariate_normal(
            np.zeros(n_data_points_exp), cov_matrix
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr is handled correctly
        std_model = 2.0
        l_corr = -1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = AdditiveSpaceCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("z")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="xy",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        space_vector = np.zeros((n_data_points_exp, 2))
        space_vector[:, 0] = dummy_data
        space_vector[:, 1] = dummy_data
        cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f_corr)
        cov_matrix += std_measurement ** 2 * np.eye(n_data_points_exp)
        expected_ll = _loglike_multivariate_normal(
            np.zeros(n_data_points_exp), cov_matrix
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_AdditiveSpaceTimeCorrelatedModelError1D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {
            "x1": 0.0,
            "x2": 1.0,
            "t": dummy_data,
            "z1": dummy_data,
            "z2": dummy_data,
        }
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "t": "t"},
                    "z2": {"x": "x2", "t": "t"},
                },
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "t": "t"},
                    "z2": {"x": "x2", "t": "t"},
                },
            },
        }

        # checks for additive_measurement_error=False
        like_model = AdditiveSpaceTimeCorrelatedModelError1D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="xt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = kron_loglike_2D_tridiag(
            np.zeros((len(problem_experiments), n_data_points_exp)),
            np.array([sensor_values["x1"], sensor_values["x2"]]),
            dummy_data,
            l_corr_space,
            std_model,
            l_corr_time,
            1,
            None,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_space is handled correctly
        std_model = 2.0
        l_corr_space = -2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_time is handled correctly
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = AdditiveSpaceTimeCorrelatedModelError1D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="xt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = kron_loglike_2D_tridiag(
            np.zeros((len(problem_experiments), n_data_points_exp)),
            np.array([sensor_values["x1"], sensor_values["x2"]]),
            dummy_data,
            l_corr_space,
            std_model,
            l_corr_time,
            1,
            std_measurement,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_AdditiveSpaceTimeCorrelatedModelError2D3D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {
            "x1": 0.0,
            "x2": 1.0,
            "y1": -1.0,
            "y2": 2.0,
            "t": dummy_data,
            "z1": dummy_data,
            "z2": dummy_data,
        }
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "y": "y1", "t": "t"},
                    "z2": {"x": "x2", "y": "y2", "t": "t"},
                },
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "y": "y1", "t": "t"},
                    "z2": {"x": "x2", "y": "y2", "t": "t"},
                },
            },
        }

        # checks for additive_measurement_error=False
        like_model = AdditiveSpaceTimeCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="xyt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        f = lambda a: correlation_function(d=a, correlation_length=l_corr_space)
        space_vector = np.array([[0.0, -1.0], [1.0, 2.0]])
        spatial_cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f)
        d0_t, d1_t = inv_cov_vec_1D(dummy_data, l_corr_time, 1.0)
        expected_ll = kron_loglike_2D(
            np.zeros((len(problem_experiments), n_data_points_exp)),
            spatial_cov_matrix,
            [d0_t, d1_t],
            None,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_space is handled correctly
        std_model = 2.0
        l_corr_space = -2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_time is handled correctly
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = AdditiveSpaceTimeCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="xyt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        f = lambda a: correlation_function(d=a, correlation_length=l_corr_space)
        space_vector = np.array([[0.0, -1.0], [1.0, 2.0]])
        spatial_cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f)
        d0_t, d1_t = inv_cov_vec_1D(dummy_data, l_corr_time, 1.0)
        expected_ll = kron_loglike_2D(
            np.zeros((len(problem_experiments), n_data_points_exp)),
            spatial_cov_matrix,
            [d0_t, d1_t],
            std_measurement,
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_MultiplicativeSpaceCorrelatedModelError2D3D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {"x": dummy_data, "y": dummy_data, "z": dummy_data}
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"z": {"x": "x", "y": "y"}},
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {"z": {"x": "x", "y": "y"}},
            },
        }

        # test for the error that is raised when the correlation variables are given
        # as scalars and vectors at the same time
        problem_experiments_error = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": {"x": 1.0, "y": dummy_data, "z": dummy_data},
                "correlation_info": {"z": {"x": "x", "y": "y"}},
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": {"x": 1.0, "y": dummy_data, "z": dummy_data},
                "correlation_info": {"z": {"x": "x", "y": "y"}},
            },
        }
        with self.assertRaises(RuntimeError):
            MultiplicativeSpaceCorrelatedModelError2D3D(
                prms_def=["std_model", "l_corr"],
                sensors=[Sensor("z")],
                experiment_names=["Exp_1", "Exp_2"],
                problem_experiments=problem_experiments_error,
                additive_measurement_error=False,
                correlation_variables="xy",
                correlation_model="exp",
                name="L1",
            )

        # checks for additive_measurement_error=False
        like_model = MultiplicativeSpaceCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("z")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="xy",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr": l_corr,
            },
        )
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        space_vector = np.zeros((n_data_points_exp, 2))
        space_vector[:, 0] = dummy_data
        space_vector[:, 1] = dummy_data
        cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f_corr)
        cov_matrix = np.multiply(
            dummy_data.reshape(-1, 1), np.multiply(dummy_data, cov_matrix)
        )
        expected_ll = _loglike_multivariate_normal(
            np.zeros(n_data_points_exp), cov_matrix
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr is handled correctly
        std_model = 2.0
        l_corr = -1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = MultiplicativeSpaceCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr"],
            sensors=[Sensor("z")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="xy",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        space_vector = np.zeros((n_data_points_exp, 2))
        space_vector[:, 0] = dummy_data
        space_vector[:, 1] = dummy_data
        cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f_corr)
        cov_matrix = np.multiply(
            dummy_data.reshape(-1, 1), np.multiply(dummy_data, cov_matrix)
        )
        cov_matrix += std_measurement ** 2 * np.eye(n_data_points_exp)
        expected_ll = _loglike_multivariate_normal(
            np.zeros(n_data_points_exp), cov_matrix
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr = 1.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z": dummy_response},
            "Exp_2": {"z": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr": l_corr,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_MultiplicativeSpaceTimeCorrelatedModelError1D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {
            "x1": 0.0,
            "x2": 1.0,
            "t": dummy_data,
            "z1": dummy_data,
            "z2": dummy_data,
        }
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "t": "t"},
                    "z2": {"x": "x2", "t": "t"},
                },
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "t": "t"},
                    "z2": {"x": "x2", "t": "t"},
                },
            },
        }

        # checks for additive_measurement_error=False
        like_model = MultiplicativeSpaceTimeCorrelatedModelError1D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="xt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        d0_x, d1_x = inv_cov_vec_1D(np.array([0.0, 1.0]), l_corr_space, std_model)
        d0_t, d1_t = inv_cov_vec_1D(dummy_data, l_corr_time, 1.0)
        expected_ll = chol_loglike_2D(
            np.zeros((2, n_data_points_exp)),
            [d0_x, d1_x],
            [d0_t, d1_t],
            1e-9,
            np.array([dummy_data, dummy_data]),
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_space is handled correctly
        std_model = 2.0
        l_corr_space = -2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_time is handled correctly
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = MultiplicativeSpaceTimeCorrelatedModelError1D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="xt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        d0_x, d1_x = inv_cov_vec_1D(np.array([0.0, 1.0]), l_corr_space, std_model)
        d0_t, d1_t = inv_cov_vec_1D(dummy_data, l_corr_time, 1.0)
        expected_ll = chol_loglike_2D(
            np.zeros((2, n_data_points_exp)),
            [d0_x, d1_x],
            [d0_t, d1_t],
            std_measurement,
            np.array([dummy_data, dummy_data]),
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

    def test_MultiplicativeSpaceTimeCorrelatedModelError2D3D(self):

        # prepare the dummy problem experiments
        n_data_points_exp = 100
        dummy_data = np.linspace(-1, 1, n_data_points_exp)
        sensor_values = {
            "x1": 0.0,
            "x2": 1.0,
            "y1": -1.0,
            "y2": 2.0,
            "t": dummy_data,
            "z1": dummy_data,
            "z2": dummy_data,
        }
        problem_experiments = {
            "Exp_1": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "y": "y1", "t": "t"},
                    "z2": {"x": "x2", "y": "y2", "t": "t"},
                },
            },
            "Exp_2": {
                "forward_model": "FwdModel",
                "sensor_values": sensor_values,
                "correlation_info": {
                    "z1": {"x": "x1", "y": "y1", "t": "t"},
                    "z2": {"x": "x2", "y": "y2", "t": "t"},
                },
            },
        }

        # checks for additive_measurement_error=False
        like_model = MultiplicativeSpaceTimeCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=False,
            correlation_variables="xyt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        space_vector = np.array([[0.0, -1.0], [1.0, 2.0]])
        f = lambda a: correlation_function(d=a, correlation_length=l_corr_space)
        spatial_cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f)
        d0_t, d1_t = inv_cov_vec_1D(dummy_data, l_corr_time, 1.0)
        expected_ll = chol_loglike_2D(
            np.zeros((len(problem_experiments), n_data_points_exp)),
            spatial_cov_matrix,
            [d0_t, d1_t],
            1e-9,
            np.array([dummy_data, dummy_data]),
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_model is handled correctly
        std_model = -2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_space is handled correctly
        std_model = 2.0
        l_corr_space = -2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative l_corr_time is handled correctly
        std_model = 2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)

        # checks for additive_measurement_error=True
        like_model = MultiplicativeSpaceTimeCorrelatedModelError2D3D(
            prms_def=["std_model", "l_corr_space", "l_corr_time"],
            sensors=[Sensor("z1"), Sensor("z2")],
            experiment_names=["Exp_1", "Exp_2"],
            problem_experiments=problem_experiments,
            additive_measurement_error=True,
            correlation_variables="xyt",
            correlation_model="exp",
            name="L1",
        )
        # the dummy-response is chosen identical to the dummy-data, resulting in zero
        # residuals; this allows a simple check if the computation works as expected
        std_model = 2.0
        std_measurement = 2.0
        l_corr_space = 2.0
        l_corr_time = 2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        space_vector = np.array([[0.0, -1.0], [1.0, 2.0]])
        f = lambda a: correlation_function(d=a, correlation_length=l_corr_space)
        spatial_cov_matrix = std_model ** 2 * correlation_matrix(space_vector, f)
        d0_t, d1_t = inv_cov_vec_1D(dummy_data, l_corr_time, 1.0)
        expected_ll = chol_loglike_2D(
            np.zeros((len(problem_experiments), n_data_points_exp)),
            spatial_cov_matrix,
            [d0_t, d1_t],
            std_measurement,
            np.array([dummy_data, dummy_data]),
        )
        self.assertAlmostEqual(computed_ll, expected_ll)

        # check now, if a negative std_measurement is handled correctly
        std_model = 2.0
        std_measurement = -2.0
        l_corr_space = 2.0
        l_corr_time = -2.0
        dummy_response = dummy_data
        model_response_dict = {
            "Exp_1": {"z1": dummy_response, "z2": dummy_response},
            "Exp_2": {"z1": dummy_response, "z2": dummy_response},
        }
        computed_ll = like_model.loglike(
            model_response_dict,
            {
                "std_model": std_model,
                "std_measurement": std_measurement,
                "l_corr_space": l_corr_space,
                "l_corr_time": l_corr_time,
            },
        )
        expected_ll = -np.infty
        self.assertAlmostEqual(computed_ll, expected_ll)


if __name__ == "__main__":
    unittest.main()
