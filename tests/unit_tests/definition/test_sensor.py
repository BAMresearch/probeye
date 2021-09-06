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
        # the PositionSensor class has only a few attributes, no methods
        sensor_name = 'Some sensor-name'
        x, y, z = 1, 2, 3
        position_sensor = PositionSensor(sensor_name, x=x, y=y, z=z)
        # check if everything is where it should be
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        for i, value in enumerate([x, y, z]):
            self.assertAlmostEqual(position_sensor.xyz[i], value)

if __name__ == "__main__":
    unittest.main()
