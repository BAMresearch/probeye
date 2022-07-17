# standard library imports
import unittest

# third party imports
from scipy import stats

# local imports
from probeye.definition.distribution import *
from probeye.inference.scipy.priors import Prior


class TestProblem(unittest.TestCase):
    def test_prior_normal(self):
        prior_normal = Prior(
            "a", ["mean_a", "std_a"], "a_normal", Normal(mean="", std="")
        )
        # check the evaluation of the log-pdf
        prms = {"a": 1.0, "mean_a": 0.0, "std_a": 1.0}
        self.assertEqual(
            stats.norm.logpdf(prms["a"], prms["mean_a"], prms["std_a"]),
            prior_normal(prms, "logpdf"),
        )
        # check the sampling-method (samples are checked one by one)
        prms = {"mean_a": 0.0, "std_a": 1.0}
        prior_samples = prior_normal.generate_samples(prms, 10, seed=1)
        sp_samples = stats.norm.rvs(
            loc=prms["mean_a"], scale=prms["std_a"], size=10, random_state=1
        )
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)
        # test multivariate version
        prior_normal = Prior(
            "a", ["mean_a", "std_a"], "a_normal", MultivariateNormal(mean="", cov="")
        )
        prms = {"mean_a": [0.0, 0.0], "cov_a": [1.0, 1.0]}
        sample = prior_normal(prms, method="rvs", use_ref_prm=False, size=10)
        self.assertEqual(len(sample), 10)
        # test requesting an invalid method
        with self.assertRaises(AttributeError):
            prior_normal(prms, method="invalid method")

    def test_prior_truncnormal(self):
        prior_truncnormal = Prior(
            "sigma",
            ["mean_sigma", "std_sigma"],
            "sigma_normal",
            TruncNormal(mean="", std="", low="", high=""),
        )
        # check the evaluation of the log-pdf
        prms = {
            "sigma": 1.0,
            "mean_sigma": 0.0,
            "std_sigma": 1.0,
            "low_sigma": 0.0,
            "high_sigma": 5.0,
        }
        self.assertEqual(
            stats.truncnorm.logpdf(
                prms["sigma"],
                a=prms["low_sigma"],
                b=prms["high_sigma"],
                loc=prms["mean_sigma"],
                scale=prms["std_sigma"],
            ),
            prior_truncnormal(prms, "logpdf"),
        )
        # check the evaluation of the mean
        mean = prior_truncnormal(prms, method="mean", use_ref_prm=False)
        self.assertAlmostEqual(
            mean,
            stats.truncnorm.mean(
                prms["low_sigma"],
                prms["high_sigma"],
                loc=prms["mean_sigma"],
                scale=prms["std_sigma"],
            ),
        )
        # check the sampling-method (samples are checked one by one)
        prms = {"mean_sigma": 0.0, "std_sigma": 1.0, "low_sigma": 0, "high_sigma": 5}
        prior_samples = prior_truncnormal.generate_samples(prms, 10, seed=1)
        sp_samples = stats.truncnorm.rvs(
            a=prms["low_sigma"],
            b=prms["high_sigma"],
            loc=prms["mean_sigma"],
            scale=prms["std_sigma"],
            size=10,
            random_state=1,
        )
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

    def test_prior_lognormal(self):
        prior_lognormal = Prior(
            "a", ["mean_a", "std_a"], "a_lognormal", LogNormal(mean="", std="")
        )
        # check the evaluation of the log-pdf
        prms = {"a": 2.0, "mean_a": 1.0, "std_a": 1.0}
        self.assertEqual(
            stats.lognorm.logpdf(
                prms["a"], scale=np.exp(prms["mean_a"]), s=prms["std_a"]
            ),
            prior_lognormal(prms, "logpdf"),
        )
        # check the evaluation of the mean
        mean = prior_lognormal(prms, method="mean", use_ref_prm=False)
        self.assertAlmostEqual(mean, stats.lognorm.mean(s=1.0, scale=np.exp(1.0)))
        # check the sampling-method (samples are checked one by one)
        prms = {"mean_a": 1.0, "std_a": 1.0}
        prior_samples = prior_lognormal.generate_samples(prms, 10, seed=1)
        sp_samples = stats.lognorm.rvs(
            1.0,  # this is scipy's shape parameter
            scale=np.exp(prms["mean_a"]),
            size=10,
            random_state=1,
        )
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

    def test_prior_uniform(self):
        prior_uniform = Prior(
            "a", ["low_a", "high_a"], "a_uniform", Uniform(low="", high="")
        )
        # check the evaluation of the log-pdf
        prms = {"a": 0.5, "low_a": 0.0, "high_a": 1.0}
        self.assertEqual(
            stats.uniform.logpdf(prms["a"], prms["low_a"], prms["high_a"]),
            prior_uniform(prms, "logpdf"),
        )
        # check the sampling-method (samples are checked one by one)
        prms = {"low_a": 0.0, "high_a": 1.0}
        prior_samples = prior_uniform.generate_samples(prms, 10, seed=1)
        sp_samples = stats.uniform.rvs(
            loc=prms["low_a"],
            scale=prms["low_a"] + prms["high_a"],
            size=10,
            random_state=1,
        )
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

    def test_prior_weibull(self):
        prior_weibull = Prior(
            "a",
            ["loc_a", "scale_a", "shape_a"],
            "a_weibull",
            Weibull(scale="", shape=""),
        )
        # check the evaluation of the log-pdf
        prms = {"a": 1.0, "scale_a": 1.0, "shape_a": 2.0}
        self.assertEqual(
            stats.weibull_min.logpdf(prms["a"], prms["shape_a"], scale=prms["scale_a"]),
            prior_weibull(prms, "logpdf"),
        )
        # check the sampling-method (samples are checked one by one)
        prms = {"scale_a": 1.0, "shape_a": 2.0}
        prior_samples = prior_weibull.generate_samples(prms, 10, seed=1)
        sp_samples = stats.weibull_min.rvs(
            prms["shape_a"],
            scale=prms["scale_a"],
            size=10,
            random_state=1,
        )
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)
        # check the evaluation of the mean
        mean = prior_weibull(prms, method="mean", use_ref_prm=False)
        self.assertAlmostEqual(mean, stats.weibull_min.mean(2, scale=1))


if __name__ == "__main__":
    unittest.main()
