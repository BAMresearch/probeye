# standard library imports
import unittest

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor


class TestProblem(unittest.TestCase):
    def test_undefined_forward_model(self):
        # check for the not-implemented-error when no response method is defined
        class ForwardModel(ForwardModelBase):
            def interface(self):
                self.parameters = "a"
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

        forward_model = ForwardModel("CoolName")
        with self.assertRaises(NotImplementedError):
            forward_model.response({})
        with self.assertRaises(NotImplementedError):
            forward_model({})

    def test_model_template_one_sensor(self):

        # define a simple model using ForwardModelBase
        class ForwardModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["a", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp):
                x = inp["x"]
                a = inp["a"]
                b = inp["b"]
                return {"y": a * x**2 + b}

        # check the __call__-method
        forward_model = ForwardModel("FunkyName")
        self.assertEqual(forward_model.sensor_names, ["x", "y"])
        prms = {"a": 1, "b": 2}
        computed_result = forward_model({**{"x": 1.0}, **prms})
        expected_result = {"y": 3.0}
        self.assertEqual(computed_result, expected_result)

        # check input/output sensor names
        self.assertEqual(forward_model.input_sensor_names, ["x"])
        self.assertEqual(forward_model.output_sensor_names, ["y"])

    def test_model_template_multiple_sensors(self):

        # define an output sensor with an offset attribute
        class OutputSensorOffset(Sensor):
            def __init__(self, name, offset):
                super().__init__(name)
                self.offset = offset

        # define a simple model using ForwardModelBase
        class ForwardModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["a", "b"]
                self.input_sensors = [Sensor("x1"), Sensor("x2")]
                self.output_sensors = [
                    OutputSensorOffset("y1", 0.1),
                    OutputSensorOffset("y2", -0.2),
                ]

            def response(self, inp):
                x1 = inp["x1"]
                x2 = inp["x2"]
                a = inp["a"]
                b = inp["b"]
                response_dict = dict()
                for os in self.output_sensors:
                    response_dict[os.name] = a * x1**2 + b * x2 + os.offset
                return response_dict

        # check the __call__-method
        forward_model = ForwardModel("SchwiftyName")
        prms = {"a": 1, "b": 2}
        computed_result = forward_model({**{"x1": 2.0, "x2": 3.0}, **prms})
        expected_result = {"y1": 10.1, "y2": 9.8}
        self.assertEqual(computed_result, expected_result)

        # check input/output sensor names
        self.assertEqual(forward_model.input_sensor_names, ["x1", "x2"])
        self.assertEqual(forward_model.output_sensor_names, ["y1", "y2"])

    def test_invalid_forward_model_definitions(self):
        class ParametersNotSet(ForwardModelBase):
            def interface(self):
                # self.parameters = [{"a": "m"}, "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        with self.assertRaises(RuntimeError):
            ParametersNotSet("Horst")

        class InputSensorsNotSet(ForwardModelBase):
            def interface(self):
                self.parameters = [{"a": "m"}, "b"]
                # self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        with self.assertRaises(RuntimeError):
            InputSensorsNotSet("Fritz")

        class OutputSensorsNotSet(ForwardModelBase):
            def interface(self):
                self.parameters = [{"a": "m"}, "b"]
                self.input_sensors = Sensor("x")
                # self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        with self.assertRaises(RuntimeError):
            OutputSensorsNotSet("Gerd")

        class InterfaceMethodNotSet(ForwardModelBase):
            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        with self.assertRaises(RuntimeError):
            InterfaceMethodNotSet("Bert")


if __name__ == "__main__":
    unittest.main()
