# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.sensor import Sensor


class TestProblem(unittest.TestCase):
    def test_sensor_no_coords_init(self):
        # so far the Sensor class has only a name attribute, no methods
        sensor_name = "Some sensor-name"
        sensor = Sensor(sensor_name)
        # check if everything is where it should be
        self.assertEqual(sensor.name, sensor_name)

    def test_sensor_with_coords_init(self):
        # check a position sensor with three coordinates via x-y-z-input
        x, y, z, sensor_name = 1, 2, 3, "Some sensor-name"
        position_sensor = Sensor(sensor_name, x=x, y=y, z=z)
        # check if everything is where it should be
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.coords.shape, (3, 1))
        self.assertEqual(position_sensor.order, ["x", "y", "z"])
        for i, value in enumerate([x, y, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with three coordinates via coords-input
        coords, sensor_name = np.array([[1], [2], [3]]), "Some sensor-name"
        position_sensor = Sensor(sensor_name, coords=coords)
        # check if everything is where it should be
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.coords.shape, (3, 1))
        self.assertEqual(position_sensor.order, ["x", "y", "z"])
        for i, value in enumerate([x, y, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with two coordinates (x and y) via x-y-input
        x, y, sensor_name = 1, 2, "Some sensor-name"
        position_sensor = Sensor(sensor_name, x=x, y=y)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.coords.shape, (2, 1))
        self.assertEqual(position_sensor.order, ["x", "y"])
        for i, value in enumerate([x, y]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with two coordinates (x, y) via coords-input
        coords, sensor_name = np.array([[1], [2]]), "Some sensor-name"
        position_sensor = Sensor(sensor_name, coords=coords)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.order, ["x", "y"])
        self.assertEqual(position_sensor.coords.shape, (2, 1))
        for i, value in enumerate([x, y]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with two coordinates (x and z) via x-z-input
        x, z, sensor_name = 1, 3, "Some sensor-name"
        position_sensor = Sensor(sensor_name, x=x, z=z)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.order, ["x", "z"])
        self.assertEqual(position_sensor.coords.shape, (2, 1))
        for i, value in enumerate([x, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with two coordinates (x, z) via coords-input
        coords, sensor_name = np.array([[1], [3]]), "Some sensor-name"
        position_sensor = Sensor(sensor_name, coords=coords, order=("x", "z"))
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.order, ["x", "z"])
        self.assertEqual(position_sensor.coords.shape, (2, 1))
        for i, value in enumerate([x, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with two coordinates (y and z) via y-z-input
        y, z, sensor_name = 2, 3, "Some sensor-name"
        position_sensor = Sensor(sensor_name, y=y, z=z)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.order, ["y", "z"])
        self.assertEqual(position_sensor.coords.shape, (2, 1))
        for i, value in enumerate([y, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with two coordinates (y, z) via coords-input
        coords, sensor_name = np.array([[2], [3]]), "Some sensor-name"
        position_sensor = Sensor(sensor_name, coords=coords, order=("y", "z"))
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.order, ["y", "z"])
        self.assertEqual(position_sensor.coords.shape, (2, 1))
        for i, value in enumerate([y, z]):
            self.assertAlmostEqual(position_sensor.coords[i], value)

        # check a position sensor with one coordinate (x) via x-input
        x, sensor_name = 1, "Some sensor-name"
        position_sensor = Sensor(sensor_name, x=x)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, x)
        self.assertEqual(position_sensor.order, ["x"])
        self.assertEqual(position_sensor.coords.shape, (1, 1))
        self.assertAlmostEqual(position_sensor.coords[0], x)

        # check a position sensor with one coordinate (x) via coords-input
        coords, sensor_name = np.array([[1]]), "Some sensor-name"
        position_sensor = Sensor(sensor_name, coords=coords)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.x, 1)
        self.assertEqual(position_sensor.order, ["x"])
        self.assertEqual(position_sensor.coords.shape, (1, 1))
        self.assertAlmostEqual(position_sensor.coords[0], 1)

        # check a position sensor with one coordinate (y) via y-input
        y, sensor_name = 2, "Some sensor-name"
        position_sensor = Sensor(sensor_name, y=y)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.y, y)
        self.assertEqual(position_sensor.order, ["y"])
        self.assertEqual(position_sensor.coords.shape, (1, 1))
        self.assertAlmostEqual(position_sensor.coords[0], y)

        # check a position sensor with one coordinate (y) via coords-input
        coords, sensor_name = np.array([[2]]), "Some sensor-name"
        # noinspection PyTypeChecker
        position_sensor = Sensor(sensor_name, coords=coords, order="y")
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.y, 2)
        self.assertEqual(position_sensor.order, ["y"])
        self.assertEqual(position_sensor.coords.shape, (1, 1))
        self.assertAlmostEqual(position_sensor.coords[0], 2)

        # check a position sensor with one coordinate (z) via z-input
        z, sensor_name = 3, "Some sensor-name"
        position_sensor = Sensor(sensor_name, z=z)
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.z, z)
        self.assertEqual(position_sensor.order, ["z"])
        self.assertEqual(position_sensor.coords.shape, (1, 1))
        self.assertAlmostEqual(position_sensor.coords[0], z)

        # check a position sensor with one coordinate (z) via coords-input
        coords, sensor_name = np.array([[3]]), "Some sensor-name"
        # noinspection PyTypeChecker
        position_sensor = Sensor(sensor_name, coords=coords, order="z")
        self.assertEqual(position_sensor.name, sensor_name)
        self.assertEqual(position_sensor.z, 3)
        self.assertEqual(position_sensor.order, ["z"])
        self.assertEqual(position_sensor.coords.shape, (1, 1))
        self.assertAlmostEqual(position_sensor.coords[0], 3)


if __name__ == "__main__":
    unittest.main()
