# third party imports
import unittest

# local imports
from probeye.definition.sensor import Sensor
from probeye.definition.sensor import PositionSensor


class TestProblem(unittest.TestCase):

    def test_sensor_init(self):
        # so far the Sensor class has only a name attribute, no methods
        sensor_name = 'Some sensor-name'
        sensor = Sensor(sensor_name)
        # check if everything is where it should be
        self.assertEqual(sensor.name, sensor_name)

    def test_position_sensor_init(self):
        # check a position sensor with three coordinates
        x, y, z, sensor_name = 1, 2, 3, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, x=x, y=y, z=z)
        # check if everything is where it should be
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.coords.shape, (3, ))
        for i, value in enumerate([x, y, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)
        # check a position sensor with two coordinates (x and y)
        x, y, sensor_name = 1, 2, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, x=x, y=y)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.coords.shape, (2,))
        for i, value in enumerate([x, y]):
            self.assertAlmostEqual(position_sensor.coords[i], value)
        # check a position sensor with two coordinates (x and z)
        x, z, sensor_name = 1, 3, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, x=x, z=z)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.coords.shape, (2,))
        for i, value in enumerate([x, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)
        # check a position sensor with two coordinates (y and z)
        y, z, sensor_name = 2, 3, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, y=y, z=z)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.coords.shape, (2,))
        for i, value in enumerate([y, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)
        # check a position sensor with one coordinate (x)
        x, sensor_name = 1, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, x=x)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.coords.shape, (1,))
        self.assertAlmostEqual(position_sensor.coords[0], x)
        # check a position sensor with one coordinate (y)
        y, sensor_name = 2, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, y=y)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.coords.shape, (1,))
        self.assertAlmostEqual(position_sensor.coords[0], y)
        # check a position sensor with one coordinate (z)
        z, sensor_name = 3, 'Some sensor-name'
        position_sensor = PositionSensor(sensor_name, z=z)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.coords.shape, (1,))
        self.assertAlmostEqual(position_sensor.coords[0], z)
        # check that an error is raised if no coordinates are given
        with self.assertRaises(RuntimeError):
            PositionSensor('Some sensor-name')

if __name__ == "__main__":
    unittest.main()
