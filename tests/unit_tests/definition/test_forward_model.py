# third party imports
import unittest

# local imports
from probeye.definition.forward_model import ForwardModelTemplate
from probeye.definition.sensor import Sensor

class TestProblem(unittest.TestCase):

    def test_model_template_one_sensor(self):

        # define a simple model using ForwardModelTemplate
        class ForwardModel(ForwardModelTemplate):
            def __call__(self, inp):
                x = inp['x']
                a = inp['a']
                b = inp['b']
                return {'y': a * x ** 2 + b}

        # check the __call__-method
        forward_model = ForwardModel(
            ['a', 'b'], [Sensor('x')], [Sensor('y')])
        prms = {'a': 1, 'b': 2}
        computed_result = forward_model({**{'x': 1.0}, **prms})
        expected_result = {'y': 3.0}
        self.assertEqual(computed_result, expected_result)

    def test_model_template_multiple_sensors(self):

        # define an output sensor with an offset attribute
        class OutputSensorOffset(Sensor):
            def __init__(self, name, offset):
                super().__init__(name)
                self.offset = offset

        # define a simple model using ForwardModelTemplate
        class ForwardModel(ForwardModelTemplate):
            def __call__(self, inp):
                x1 = inp['x1']
                x2 = inp['x2']
                a = inp['a']
                b = inp['b']
                response_dict = dict()
                for os in self.output_sensors:
                    response_dict[os.name] = a * x1**2 + b * x2 + os.offset
                return response_dict

        # check the __call__-method
        is1, is2 = Sensor('x1'), Sensor('x2')
        os1, os2 = OutputSensorOffset('y1', 0.1), OutputSensorOffset('y2', -0.2)
        forward_model = ForwardModel(['a', 'b'], [is1, is2], [os1, os2])
        prms = {'a': 1, 'b': 2}
        computed_result = forward_model({**{'x1': 2.0, 'x2': 3.0}, **prms})
        expected_result = {'y1': 10.1, 'y2': 9.8}
        self.assertEqual(computed_result, expected_result)

if __name__ == "__main__":
    unittest.main()
