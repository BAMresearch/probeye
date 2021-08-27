# third party imports
import unittest

# local imports
from probeye.definition.forward_model import ModelTemplate
from probeye.definition.forward_model import Sensor

class TestProblem(unittest.TestCase):

    def test_model_template_one_sensor(self):

        # define a simple model using ModelTemplate
        class ForwardModel(ModelTemplate):
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

        # check the jacobian-method
        forward_model = ForwardModel(
            ['a', 'b'], [Sensor('x')], [Sensor('y')])
        prms = {'a': 1, 'b': 2}
        computed_result = forward_model.jacobian({'x': 3}, prms)
        expected_result = {'a': {'y': 9}, 'b': {'y': 1}}
        for prm_name in expected_result.keys():
            self.assertAlmostEqual(computed_result[prm_name]['y'],
                                   expected_result[prm_name]['y'])

        # check the error/error_function-method
        forward_model = ForwardModel(
            ['a', 'b'], [Sensor('x')], [Sensor('y')])
        prms = {'a': 1, 'b': 2}
        experiments = {'Experiment_1': {'x': 1, 'y': 3.1},
                       'Experiment_2': {'x': 2, 'y': 5.8}}
        computed_result = forward_model.error(prms, experiments)
        expected_result = {'y': [-0.1, 0.2]}
        for i in range(len(expected_result)):
            self.assertAlmostEqual(computed_result['y'][i],
                                   expected_result['y'][i])

    def test_model_template_multiple_sensors(self):

        # define an output sensor with an offset attribute
        class OutputSensorOffset(Sensor):
            def __init__(self, name, offset):
                super().__init__(name)
                self.offset = offset

        # define a simple model using ModelTemplate
        class ForwardModel(ModelTemplate):
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

        # check the jacobian-method
        is1, is2 = Sensor('x1'), Sensor('x2')
        os1, os2 = OutputSensorOffset('y1', 0.1), OutputSensorOffset('y2', -0.2)
        forward_model = ForwardModel(['a', 'b'], [is1, is2], [os1, os2])
        prms = {'a': 1, 'b': 2}
        computed_result = forward_model.jacobian({'x1': 2.0, 'x2': 3.0}, prms)
        expected_result = {'a': {'y1': 4, 'y2': 4}, 'b': {'y1': 3, 'y2': 3}}
        for prm_name in expected_result.keys():
            for os_name in expected_result[prm_name].keys():
                self.assertAlmostEqual(computed_result[prm_name][os_name],
                                       expected_result[prm_name][os_name])

        # check the error/error_function-method
        is1, is2 = Sensor('x1'), Sensor('x2')
        os1, os2 = OutputSensorOffset('y1', 0.1), OutputSensorOffset('y2', -0.2)
        forward_model = ForwardModel(['a', 'b'], [is1, is2], [os1, os2])
        prms = {'a': 1, 'b': 2}
        experiments = {'Exp_1': {'x1': 2, 'x2': 3, 'y1': 10.4, 'y2': 9.7},
                       'Exp_2': {'x1': 3, 'x2': 4, 'y1': 17.8, 'y2': 17.4}}
        computed_result = forward_model.error(prms, experiments)
        expected_result = {'y1': [-0.3, -0.7], 'y2': [0.1, -0.6]}
        for i in range(len(expected_result)):
            for os_name in expected_result.keys():
                self.assertAlmostEqual(computed_result[os_name][i],
                                       expected_result[os_name][i])

if __name__ == "__main__":
    unittest.main()
