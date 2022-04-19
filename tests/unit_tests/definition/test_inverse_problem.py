# standard library
import unittest
import io
import sys

# third party imports
import numpy as np

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel


class TestProblem(unittest.TestCase):
    def test_properties(self):
        p = InverseProblem("TestProblem")
        # add first latent parameter; n_latent_prms should be 1
        p.add_parameter("a", "model", prior=("normal", {"mean": 0.0, "std": 1.0}))
        self.assertEqual(p.n_latent_prms, 1)
        self.assertEqual(p.latent_prms_dims, [1])
        # add second latent parameter; n_latent_prms should be 2
        p.add_parameter("b", "likelihood", prior=("normal", {"mean": 0.0, "std": 1.0}))
        self.assertEqual(p.n_latent_prms, 2)
        self.assertEqual(p.latent_prms_dims, [1, 1])
        # check the different properties
        self.assertEqual(p.n_prms, 6)
        self.assertEqual(set(p.prms), {"mean_a", "std_a", "a", "mean_b", "std_b", "b"})
        self.assertEqual(p.n_latent_prms, 2)
        self.assertEqual(set(p.latent_prms), {"a", "b"})
        self.assertEqual(p.n_constant_prms, 4)
        self.assertEqual(set(p.constant_prms), {"mean_a", "std_a", "mean_b", "std_b"})
        self.assertEqual(p.n_model_prms, 1)
        self.assertEqual(set(p.model_prms), {"a"})
        self.assertEqual(p.n_prior_prms, 4)
        self.assertEqual(set(p.prior_prms), {"mean_a", "std_a", "mean_b", "std_b"})
        self.assertEqual(p.n_likelihood_prms, 1)
        self.assertEqual(set(p.likelihood_prms), {"b"})

    def test_info(self):
        # simply check that no errors occur when info-method is called; the output that
        # is usually printed is redirected
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", const=1.0)
        p.add_parameter("b", "model", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter(
            "sigma_model", "likelihood", prior=("normal", {"mean": 0, "std": 1})
        )
        sys.stdout = io.StringIO()
        # try out different options
        p.info(tablefmt="presto", check_consistency=False)
        _ = p.info(tablefmt="presto", check_consistency=False)
        p.info(tablefmt="presto", check_consistency=False, print_header=True)
        p.info(tablefmt="plain", check_consistency=False)
        sys.stdout = sys.__stdout__  # reset printout to console
        with self.assertRaises(AssertionError):
            # the problem is not consistent yet (e.g. no forward model defined yet), so
            # the consistency_check will raise an error
            p.info(tablefmt="presto", check_consistency=True)
        # now add the remaining stuff to make to problem consistent

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "b"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma_model")

        test_model = FwdModel("TestModel")
        p.add_forward_model(test_model)
        p.add_experiment(
            "Experiment_1", sensor_values={"x": 1, "y": 1}, fwd_model_name="TestModel"
        )
        p.add_likelihood_model(
            GaussianLikelihoodModel("sigma_model", experiment_name="Experiment_1")
        )
        sys.stdout = io.StringIO()
        # now, the consistency_check should not raise an error
        p.info(tablefmt="presto", check_consistency=True)
        sys.stdout = sys.__stdout__  # reset printout to console

    def test_str(self):
        # set up a consistent problem and print it
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", const=1.0)
        p.add_parameter("b", "model", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter(
            "sigma_model", "likelihood", prior=("normal", {"mean": 0, "std": 1})
        )

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "b"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma_model")

        test_model = FwdModel("TestModel")
        p.add_forward_model(test_model)

        p.add_experiment(
            "Experiment_1", sensor_values={"x": 1, "y": 1}, fwd_model_name="TestModel"
        )
        p.add_likelihood_model(
            GaussianLikelihoodModel("sigma_model", experiment_name="Experiment_1")
        )
        sys.stdout = io.StringIO()  # redirect output to console
        print(p)
        sys.stdout = sys.__stdout__  # reset printout to console

    def test_add_parameter(self):
        p = InverseProblem("TestProblem")
        # check valid use cases for constant parameters
        p.add_parameter("c", "model", const=1.0, info="info", tex=r"$c$")
        p.add_parameter("mean_a", "prior", const=1.0, info="info", tex=r"$mean_a$")
        p.add_parameter(
            "sigma_1", "likelihood", const=1.0, info="info", tex=r"$\sigma_1$"
        )
        # check valid use cases for latent parameters
        p.add_parameter(
            "b",
            "model",
            prior=("normal", {"mean": 0, "std": 1}),
            info="info",
            tex="$b$",
        )
        p.add_parameter(
            "std_a",
            "prior",
            info="info",
            tex="$b$",
            prior=("uniform", {"low": 0.1, "high": 2.0}),
        )
        p.add_parameter(
            "sigma_2",
            "likelihood",
            info="info",
            tex=r"$\sigma_2$",
            prior=("uniform", {"low": 0.1, "high": 2.0}),
        )
        p.add_parameter(
            "a",
            "model",
            info="info",
            tex="$a$",
            prior=("normal", {"mean": "mean_a", "std": "std_a"}),
        )
        p.add_parameter("d", "model")  # latent param. with uninformative prior
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # adding a parameter with wrong type
            p.add_parameter("d", "wrong_type", const=1.0)
        with self.assertRaises(RuntimeError):
            # adding a parameter with both const and prior given
            p.add_parameter(
                "a", "model", const=1.0, prior=("uniform", {"low": 0.1, "high": 2.0})
            )
        with self.assertRaises(RuntimeError):
            # adding a parameter with a name that was used before
            p.add_parameter("c", "likelihood", const=2.0, info="info", tex=r"$c$")
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            # noinspection PyTypeChecker
            p.add_parameter(
                "eps",
                "model",
                info="info",
                tex=r"$\epsilon$",
                prior="Wrong prior format",
            )
        with self.assertRaises(RuntimeError):
            # adding a parameter with an invalid prior
            p.add_parameter(
                "eps",
                "model",
                info="info",
                tex=r"$\epsilon$",
                prior=[{"low": 0.1, "high": 2.0}],
            )
        with self.assertRaises(RuntimeError):
            # adding a parameter with an invalid prior
            p.add_parameter(
                "eps",
                "model",
                info="info",
                tex=r"$\epsilon$",
                prior=("uniform", {"low": 0.1, "high": 2.0}, "3"),
            )
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter(
                "eps",
                "model",
                info="info",
                tex=r"$\epsilon$",
                prior=(None, {"low": 0.1, "high": 2.0}),
            )
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter(
                "eps",
                "model",
                info="info",
                tex=r"$\epsilon$",
                prior=("uniform", [{"low": 0.1}, {"high": 2.0}]),
            )
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter(
                "eps",
                "model",
                info="info",
                tex=r"$\epsilon$",
                prior=("uniform", {"low": None, "high": 2.0}),
            )

    def test_remove_parameter(self):
        # set up a problem with some parameters so we can remove them
        p = InverseProblem("TestProblem")
        p.add_parameter("c", "model", const=1.0, info="info", tex=r"$c$")
        p.add_parameter("mean_a", "prior", const=1.0, info="info", tex=r"$mean_a$")
        p.add_parameter(
            "sigma_model", "likelihood", const=1.0, info="info", tex=r"$\sigma$"
        )
        p.add_parameter(
            "std_a",
            "prior",
            info="info",
            tex="$b$",
            prior=("uniform", {"low": 0.1, "high": 2.0}),
        )
        p.add_parameter(
            "sigma",
            "likelihood",
            info="info",
            tex=r"$\sigma$",
            prior=("uniform", {"low": 0.1, "high": 2.0}),
        )
        p.add_parameter(
            "a",
            "model",
            info="info",
            tex="$a$",
            prior=("normal", {"mean": "mean_a", "std": "std_a"}),
        )
        # check removing a constant parameter
        self.assertEqual(
            set(p.constant_prms),
            {
                "c",
                "mean_a",
                "sigma_model",
                "low_std_a",
                "high_std_a",
                "low_sigma",
                "high_sigma",
            },
        )
        p.remove_parameter("c")  # <-- this is where the removal happens
        self.assertEqual(
            set(p.constant_prms),
            {
                "mean_a",
                "sigma_model",
                "low_std_a",
                "high_std_a",
                "low_sigma",
                "high_sigma",
            },
        )
        # check removing a latent parameter; note that removing a latent parameter leads
        # to the removal of all its prior parameters
        self.assertEqual(set(p.latent_prms), {"std_a", "sigma", "a"})
        # check that indexes are as expected
        self.assertEqual(p.parameters["std_a"].index, 0)
        self.assertEqual(p.parameters["sigma"].index, 1)
        self.assertEqual(p.parameters["a"].index, 2)
        p.remove_parameter("a")  # <-- this is where the removal happens
        # check that all latent parameters associated with 'a' are gone
        self.assertEqual(set(p.latent_prms), {"sigma"})
        # check if the re-indexing worked
        self.assertEqual(p.parameters["sigma"].index, 0)
        # check that all constant prior-parameters associated with 'a' are gone
        self.assertEqual(
            set(p.constant_prms), {"sigma_model", "low_sigma", "high_sigma"}
        )
        # check that the removal of non-existing parameters results in an error
        with self.assertRaises(RuntimeError):
            p.remove_parameter("undefined")

    def test_check_if_parameter_exists(self):
        # check confirming existing parameter
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))
        p.parameters.confirm_that_parameter_exists("a")
        # check RuntimeError for non-existing parameter
        with self.assertRaises(RuntimeError):
            p.parameters.confirm_that_parameter_exists("b")

    def test_change_parameter_role(self):
        p = InverseProblem("TestProblem")
        # check change of role from latent to constant parameter
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))
        self.assertEqual(set(p.latent_prms), {"a"})
        self.assertEqual(set(p.constant_prms), {"mean_a", "std_a"})
        self.assertEqual(set(p.prms), {"a", "mean_a", "std_a"})
        p.change_parameter_role("a", const=1.0)  # <-- here the role changes
        with self.assertRaises(RuntimeError):
            # trying to change a constant to a constant
            p.change_parameter_role("a", const=1.0)
        self.assertEqual(set(p.latent_prms), set())
        self.assertEqual(set(p.constant_prms), {"a"})
        self.assertEqual(set(p.prms), {"a"})
        # check change of role from latent to constant parameter
        p.change_parameter_role("a", prior=("normal", {"mean": 0, "std": 1}))
        self.assertEqual(set(p.latent_prms), {"a"})
        self.assertEqual(set(p.constant_prms), {"mean_a", "std_a"})
        self.assertEqual(set(p.prms), {"a", "mean_a", "std_a"})
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # stated parameter does not exist
            p.change_parameter_role("undefined", const=1.0)
        with self.assertRaises(RuntimeError):
            # neither const nor prior are given
            p.change_parameter_role("a")
        with self.assertRaises(RuntimeError):
            # both const and prior are given
            p.change_parameter_role(
                "a", const=1.0, prior=("normal", {"mean": 0, "std": 1})
            )
        with self.assertRaises(RuntimeError):
            # change to role the parameter already has
            p.change_parameter_role("a", prior=("normal", {"mean": 0, "std": 1}))

    def test_change_parameter_type(self):
        p = InverseProblem("TestProblem")
        # check change of type from 'model' to 'likelihood'
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))
        self.assertEqual(p.parameters["a"].type, "model")
        p.change_parameter_type("a", "likelihood")
        self.assertEqual(p.parameters["a"].type, "likelihood")
        # check change of type from 'likelihood' to invalid type 'INVALID'
        with self.assertRaises(ValueError):
            p.change_parameter_type("a", "INVALID")

    def test_change_parameter_info(self):
        p = InverseProblem("TestProblem")
        # simple check that the change works and has the expected effect
        p.add_parameter(
            "a",
            "model",
            prior=("normal", {"mean": 0, "std": 1}),
            info="Info",
            tex="$a$",
        )
        new_info, new_tex = "New info", r"$\hat{a}$"
        p.change_parameter_info("a", new_info, new_tex=new_tex)
        self.assertEqual(p.parameters["a"].info, new_info)
        self.assertEqual(p.parameters["a"].tex, new_tex)
        # check that you can just change either the info- or the tex-string
        new_info, new_tex = "Even newer info", r"$\hat{a}_{wow}$"
        p.change_parameter_info("a", new_info=new_info)
        p.change_parameter_info("a", new_tex=new_tex)
        self.assertEqual(p.parameters["a"].info, new_info)
        self.assertEqual(p.parameters["a"].tex, new_tex)
        # check call for non-existing parameter
        with self.assertRaises(RuntimeError):
            p.change_parameter_info("b", new_info, new_tex=new_tex)

    def test_change_constant(self):
        p = InverseProblem("TestProblem")
        # simple check that the change works and has the expected effect
        p.add_parameter("a", "model", const=0.0)
        p.add_parameter("b", "model", prior=("normal", {"mean": 0, "std": 1}))
        new_const = 1.0
        p.change_constant("a", new_const)
        self.assertEqual(p.parameters["a"].value, new_const)
        # check for invalid input arguments
        with self.assertRaises(RuntimeError):
            # given parameter does not exist
            p.change_constant("undefined", 2.0)
        with self.assertRaises(RuntimeError):
            # given parameter is not a constant parameter
            p.change_constant("b", 2.0)

    def test_check_problem_consistency(self):
        p = InverseProblem("TestProblem")
        # check if the method identifies inconsistent problems
        with self.assertRaises(AssertionError):
            # nothing is defined at this point
            p.check_problem_consistency()
        # define some parameters that do not require priors
        p.add_parameter("mean_a", "prior", const=0.0)
        p.add_parameter("std_a", "prior", const=1.0)
        with self.assertRaises(AssertionError):
            # no priors defined yet
            p.check_problem_consistency()
        # add a prior by defining a latent parameter
        p.add_parameter(
            "a",
            "model",
            info="Some model parameter",
            tex="$a$",
            prior=("normal", {"mean": "mean_a", "std": "std_a"}),
        )
        with self.assertRaises(AssertionError):
            # no forward models defined yet
            p.check_problem_consistency()

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma_model")

        test_model = FwdModel("TestModel")
        p.add_forward_model(test_model)
        with self.assertRaises(AssertionError):
            # no noise models defined yet
            p.check_problem_consistency()
        # add an experiment
        p.add_experiment(
            "Experiment_1", sensor_values={"x": 1, "y": 1}, fwd_model_name="TestModel"
        )
        # add an unused experiment
        p.add_experiment(
            "Experiment_unused",
            sensor_values={"y": 1, "x": 1},
            fwd_model_name="TestModel",
        )
        # add a noise model
        p.add_parameter("s", "likelihood", const=1.0)
        like_model = GaussianLikelihoodModel("s", experiment_name="Experiment_1")
        p.add_likelihood_model(like_model)
        # now the problem should be consistent
        p.check_problem_consistency()

    def test_add_experiment(self):
        # check correct use
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma_model")

        test_model = FwdModel("TestModel")
        p.add_forward_model(test_model)

        p.add_experiment(
            "Experiment_1", sensor_values={"x": 1, "y": 1}, fwd_model_name="TestModel"
        )
        # the experiment may contain more sensors than needed by forward model
        p.add_experiment(
            "Experiment_2",
            sensor_values={"x": 1, "y": 1, "z": 1},
            fwd_model_name="TestModel",
        )
        # check that scalar sensor_values are not transformed to numpy arrays
        x_in_p = p.experiments["Experiment_2"]["sensor_values"]["x"]
        self.assertEqual(type(x_in_p), type(1))
        # check invalid input arguments
        with self.assertRaises(TypeError):
            # wrong sensor_values type
            # noinspection PyTypeChecker
            p.add_experiment(
                "Experiment_3",
                sensor_values=[("x", 1), ("y", 2)],
                fwd_model_name="TestModel",
            )
        with self.assertRaises(TypeError):
            # wrong fwd_model_name type
            # noinspection PyTypeChecker
            p.add_experiment(
                "Experiment_3", sensor_values={"x": 1, "y": 1}, fwd_model_name=1.2
            )
        with self.assertRaises(RuntimeError):
            # referencing non-existing forward model
            p.add_experiment(
                "Experiment_3",
                sensor_values={"x": 1, "y": 1},
                fwd_model_name="ThisModelDoesNotExist",
            )
        with self.assertRaises(RuntimeError):
            # forward model's input sensor not provided by experiment
            p.add_experiment(
                "Experiment_3", sensor_values={"y": 1}, fwd_model_name="TestModel"
            )
        with self.assertRaises(RuntimeError):
            # forward model's output sensor not provided by experiment
            p.add_experiment(
                "Experiment_3", sensor_values={"x": 1}, fwd_model_name="TestModel"
            )
        # check that sensor_value lists are transformed to numpy arrays
        p.add_experiment(
            "Experiment_3",
            fwd_model_name="TestModel",
            sensor_values={"x": [1, 2], "y": [2, 3]},
        )
        x_in_p = p.experiments["Experiment_3"]["sensor_values"]["x"]
        y_in_p = p.experiments["Experiment_3"]["sensor_values"]["y"]
        self.assertEqual(type(x_in_p), np.ndarray)
        self.assertEqual(type(y_in_p), np.ndarray)
        # check adding the same experiment again
        p.add_experiment(
            "Experiment_1", sensor_values={"x": 1, "y": 1}, fwd_model_name="TestModel"
        )
        # add an experiment the correlation_info of which has a key that does not
        # appear in the experiment's sensor_value
        with self.assertRaises(RuntimeError):
            p.add_experiment(
                "Experiment_CORR_1",
                sensor_values={"x": 1, "y": 1},
                fwd_model_name="TestModel",
                correlation_info="g:x",  # g is the intended problem here
            )
        # add an experiment the correlation_info of which has a key that does not
        # appear in the forward model's output sensor names
        with self.assertRaises(RuntimeError):
            p.add_experiment(
                "Experiment_CORR_2",
                sensor_values={"x": 1, "y": 1, "z": 1},
                fwd_model_name="TestModel",
                correlation_info="z:x",  # z is the intended problem here
            )
        # add an experiment the correlation_info of which has a wrong format
        with self.assertRaises(TypeError):
            p.add_experiment(
                "Experiment_CORR_2",
                sensor_values={"x": 1, "y": 1, "z": 1},
                fwd_model_name="TestModel",
                correlation_info={"y": "x"},  # correct would be {"y": {"x": "x"}}
            )
        # add an experiment the correlation_info of which has a wrong format
        with self.assertRaises(RuntimeError):
            p.add_experiment(
                "Experiment_CORR_2",
                sensor_values={"x": 1, "y": 1, "z": 1},
                fwd_model_name="TestModel",
                correlation_info={"y": {"x": "u"}},  # u is not in the sensor_values
            )
        # add an experiment with an invalid sensor_value type
        with self.assertRaises(ValueError):
            p.add_experiment(
                "Exp_invalid_sensor_value_type",
                sensor_values={"x": 1, "y": "1"},
                fwd_model_name="TestModel",
            )
        # add an experiment with an multidimensional array as a sensor value
        with self.assertRaises(ValueError):
            p.add_experiment(
                "Exp_invalid_array_shape",
                sensor_values={"x": 1, "y": np.ones((2, 2))},
                fwd_model_name="TestModel",
            )
        # add an experiment with a scalar that is given in array format
        with self.assertRaises(ValueError):
            p.add_experiment(
                "Exp_invalid_scalar_as_array",
                sensor_values={"x": 1, "y": np.array([1.0])},
                fwd_model_name="TestModel",
            )

    def test_get_parameters(self):
        # check a simple use case
        p = InverseProblem("TestProblem")
        p.add_parameter("c", "model", const=1.0)
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter("b", "model", prior=("normal", {"mean": 0, "std": 1}))
        a_value, b_value = 3.1, 14.7
        prms_def = {"b": "b", "a": "a", "mean_b": "mean_b", "c": "c"}
        computed_result = p.get_parameters(np.array([a_value, b_value]), prms_def)
        expected_result = {"b": b_value, "a": a_value, "mean_b": 0, "c": 1.0}
        self.assertEqual(computed_result, expected_result)

    def test_get_experiment_names(self):
        # prepare for checks
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))

        class FwdModel1(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = [Sensor("y1"), Sensor("y2")]

        class FwdModel2(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = [Sensor("z1"), Sensor("z2")]

        test_model_1 = FwdModel1("TestModel_1")
        p.add_forward_model(test_model_1)

        test_model_2 = FwdModel2("TestModel_2")
        p.add_forward_model(test_model_2)

        # define experiment_names to each forward model
        p.add_experiment(
            "Experiment_Y1",
            sensor_values={"x": 1, "y1": 1, "y2": 1},
            fwd_model_name="TestModel_1",
        )
        p.add_experiment(
            "Experiment_Y2",
            sensor_values={"x": 2, "y1": 2, "y2": 2},
            fwd_model_name="TestModel_1",
        )
        p.add_experiment(
            "Experiment_Z1",
            sensor_values={"x": -1, "z1": -1, "z2": -1},
            fwd_model_name="TestModel_2",
        )
        p.add_experiment(
            "Experiment_Z2",
            sensor_values={"x": -2, "z1": -2, "z2": -2},
            fwd_model_name="TestModel_2",
        )
        # get experiment_names for specific forward model
        computed_result = p.get_experiment_names(forward_model_names="TestModel_1")
        expected_result = ["Experiment_Y1", "Experiment_Y2"]
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names for specific forward model
        computed_result = p.get_experiment_names(forward_model_names="TestModel_2")
        expected_result = ["Experiment_Z1", "Experiment_Z2"]
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names for both forward models
        computed_result = p.get_experiment_names(
            forward_model_names=["TestModel_1", "TestModel_2"]
        )
        expected_result = [
            "Experiment_Y1",
            "Experiment_Y2",
            "Experiment_Z1",
            "Experiment_Z2",
        ]
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via output sensors
        computed_result = p.get_experiment_names(sensor_names=["y1", "y2"])
        expected_result = ["Experiment_Y1", "Experiment_Y2"]
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via output sensors
        computed_result = p.get_experiment_names(sensor_names=["z1", "z2"])
        expected_result = ["Experiment_Z1", "Experiment_Z2"]
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via output sensors
        computed_result = p.get_experiment_names(sensor_names=["z1", "y1", "z2"])
        expected_result = []
        self.assertEqual(computed_result, expected_result)
        # get experiment_names via both forward model name and sensors
        computed_result = p.get_experiment_names(
            sensor_names=["y1", "y2"], forward_model_names="TestModel_1"
        )
        expected_result = ["Experiment_Y1", "Experiment_Y2"]
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via both forward model name and sensors
        computed_result = p.get_experiment_names(
            sensor_names=["y1", "y2"], forward_model_names="TestModel_2"
        )
        expected_result = []
        self.assertEqual(computed_result, expected_result)
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # neither forward_model_names nor sensor_names given
            p.get_experiment_names()

    def test_get_theta_names(self):
        # check some simple use cases
        p = InverseProblem("TestProblem")
        p.add_parameter(
            "a", "model", prior=("normal", {"mean": 0, "std": 1}), tex="$a$"
        )
        p.add_parameter(
            "b", "model", prior=("normal", {"mean": 0, "std": 1}), tex="$b$"
        )
        computed_result = p.get_theta_names(tex=False)
        expected_result = ["a", "b"]
        self.assertEqual(computed_result, expected_result)
        computed_result = p.get_theta_names(tex=True)
        expected_result = ["$a$", "$b$"]
        self.assertEqual(computed_result, expected_result)
        # check use case with vector-valued parameter
        p = InverseProblem("TestProblem")
        p.add_parameter(
            "a",
            "model",
            dim=2,
            tex="$a$",
            prior=("normal", {"mean": [0, 0], "cov": [[1, 0], [0, 1]]}),
        )
        p.add_parameter(
            "b", "model", tex="$b$", prior=("normal", {"mean": 0, "std": 1})
        )
        computed_result = p.get_theta_names(tex=False, components=False)
        expected_result = ["a", "b"]
        self.assertEqual(computed_result, expected_result)
        computed_result = p.get_theta_names(tex=False, components=True)
        expected_result = ["a_1", "a_2", "b"]
        self.assertEqual(computed_result, expected_result)
        computed_result = p.get_theta_names(tex=True, components=False)
        expected_result = ["$a$", "$b$"]
        self.assertEqual(computed_result, expected_result)
        computed_result = p.get_theta_names(tex=True, components=True)
        expected_result = ["$a_1$", "$a_2$", "$b$"]
        self.assertEqual(computed_result, expected_result)

    def test_theta_explanation(self):
        # simply check that no errors occur when theta_explanation is called; the output
        # that is usually printed is redirected
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", const=1.0)
        p.add_parameter("b", "model", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter(
            "sigma_model", "likelihood", prior=("normal", {"mean": 0, "std": 1})
        )
        # check the model_consistency flag
        with self.assertRaises(AssertionError):
            # the model is not consistent
            p.theta_explanation(check_consistency=True)

    def test_add_forward_model(self):
        # check correct use
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma_model")

        test_model = FwdModel("TestModel")
        p.add_forward_model(test_model)

        # check for invalid input arguments
        with self.assertRaises(RuntimeError):
            # a given parameter of the forward model has not been added yet
            class FwdModel2(ForwardModelBase):
                def interface(self):
                    self.parameters = "b"
                    self.input_sensors = Sensor("x")
                    self.output_sensors = Sensor("y", std_model="sigma_model")

            test_model_2 = FwdModel2("TestModel_2")
            p.add_forward_model(test_model_2)
        with self.assertRaises(RuntimeError):
            # add a forward model with the same name
            p.add_forward_model(test_model)

    def test_add_likelihood_model(self):
        # check correct use
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter("s1", "likelihood", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter("s2", "likelihood", prior=("normal", {"mean": 0, "std": 1}))
        p.add_parameter("s3", "likelihood", prior=("normal", {"mean": 0, "std": 1}))

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = [Sensor("y1"), Sensor("y2")]

        p.add_forward_model(FwdModel("TestModel"))
        # try to add a likelihood model before adding experiments (not allowed)
        with self.assertRaises(RuntimeError):
            p.add_likelihood_model(GaussianLikelihoodModel("s1", "Exp"))
        p.add_experiment(
            "Exp", fwd_model_name="TestModel", sensor_values={"x": 0, "y1": 0, "y2": 0}
        )
        p.add_likelihood_model(GaussianLikelihoodModel("s1", experiment_name="Exp"))
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # the given noise model parameter has not been defined
            p.add_likelihood_model(
                GaussianLikelihoodModel(
                    "not_existing_parameter",
                    "Exp",
                )
            )

    def test_transform_experimental_data(self):
        # check correct use
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

        p.add_forward_model(FwdModel("TestModel"))
        p.add_experiment(
            "Experiment_1",
            fwd_model_name="TestModel",
            sensor_values={"x": [1, 2], "y": [1, 3]},
        )
        # apply a simple power function to the experimental data
        p_copy = p.transform_experimental_data(
            func=np.power, args=([2, 3],), where=True
        )
        x_computed = p_copy.experiments["Experiment_1"]["sensor_values"]["x"]
        x_expected = np.array([1, 8])
        self.assertTrue(
            np.allclose(x_computed, x_expected) and x_computed.shape == x_expected.shape
        )
        y_computed = p_copy.experiments["Experiment_1"]["sensor_values"]["y"]
        y_expected = np.array([1, 27])
        self.assertTrue(
            np.allclose(y_computed, y_expected) and y_computed.shape == y_expected.shape
        )


if __name__ == "__main__":
    unittest.main()
