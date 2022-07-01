# standard library imports
import unittest

# local imports
from probeye.definition.correlation_model import CorrelationModel


class TestProblem(unittest.TestCase):
    def test_correlation_model(self):
        corr_model = CorrelationModel(x__y__z=["l_x", "l_y", "l_z"], t="l_t")
        self.assertEqual(corr_model.correlation_variables, [("x", "y", "z"), "t"])
        self.assertEqual(corr_model.parameters, ["l_x", "l_y", "l_z", "l_t"])
        self.assertEqual(
            corr_model.corr_dict, {("x", "y", "z"): ["l_x", "l_y", "l_z"], "t": "l_t"}
        )
        with self.assertRaises(ValueError):
            CorrelationModel(t=1.0)


if __name__ == "__main__":
    unittest.main()
