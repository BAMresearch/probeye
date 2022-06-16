# standard library imports
import unittest

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.distribution import Normal
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot


class TestProblem(unittest.TestCase):
    def test_not_implemented_pair_plot(self):
        # check for currently not implemented plotting frameworks
        p = InverseProblem("TestProblem")
        p.add_parameter("a", "model", prior=Normal(mean=0, std=1))
        # only one parameter given (no error, just warning)
        create_pair_plot(None, p, plot_with="arviz")
        # add another parameter to have a valid setup
        p.add_parameter("b", "model", prior=Normal(mean=0, std=1))
        with self.assertRaises(NotImplementedError):
            create_pair_plot(None, p, plot_with="seaborn")
        with self.assertRaises(NotImplementedError):
            create_pair_plot(None, p, plot_with="matplotlib")
        with self.assertRaises(RuntimeError):
            create_pair_plot(None, p, plot_with="other package")

    def test_not_implemented_create_posterior_plot(self):
        # check for currently not implemented plotting frameworks
        p = InverseProblem("TestProblem")
        with self.assertRaises(NotImplementedError):
            create_posterior_plot(None, p, plot_with="seaborn")
        with self.assertRaises(NotImplementedError):
            create_posterior_plot(None, p, plot_with="matplotlib")
        with self.assertRaises(RuntimeError):
            create_posterior_plot(None, p, plot_with="other package")

    def test_not_implemented_create_trace_plot(self):
        # check for currently not implemented plotting frameworks
        p = InverseProblem("TestProblem")
        with self.assertRaises(NotImplementedError):
            create_trace_plot(None, p, plot_with="seaborn")
        with self.assertRaises(NotImplementedError):
            create_trace_plot(None, p, plot_with="matplotlib")
        with self.assertRaises(RuntimeError):
            create_trace_plot(None, p, plot_with="other package")


if __name__ == "__main__":
    unittest.main()
