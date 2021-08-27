# third party imports
import unittest
from scipy import stats

# local imports
from probeye.definition.prior import PriorTemplate
from probeye.definition.prior import PriorNormal
from probeye.definition.prior import PriorLognormal
from probeye.definition.prior import PriorUniform
from probeye.definition.prior import PriorWeibull


class TestProblem(unittest.TestCase):

    def test_prior_template(self):
        prior_template = PriorTemplate('a', ['loc_a', 'scale_a'],
                                       'a_normal', 'normal distribution')
        # check that the attributes are wired correctly
        self.assertEqual(prior_template.ref_prm, 'a')
        self.assertEqual(prior_template.name, 'a_normal')
        self.assertEqual(prior_template.prior_type, 'normal distribution')
        self.assertEqual(prior_template.prms_def,
                         {'a': 'a', 'loc_a': 'loc_a', 'scale_a': 'scale_a'})
        self.assertEqual(prior_template.prms_def_no_ref,
                         {'loc_a': 'loc_a', 'scale_a': 'scale_a'})
        # check that the __str__-method works
        self.assertTrue(len(prior_template.__str__()) > 0)

    def test_prior_normal(self):
        prior_normal = PriorNormal('a', ['loc_a', 'scale_a'], 'a_normal')
        # check the evaluation of the log-pdf
        prms = {'a': 1.0, 'loc_a': 0.0, 'scale_a': 1.0}
        self.assertEqual(
            stats.norm.logpdf(prms['a'], prms['loc_a'], prms['scale_a']),
            prior_normal(prms, 'logpdf'))
        # check the sampling-method (samples are checked one by one)
        prms = {'loc_a': 0.0, 'scale_a': 1.0}
        prior_samples = prior_normal.generate_samples(prms, 10, seed=1)
        sp_samples = stats.norm.rvs(loc=prms['loc_a'], scale=prms['scale_a'],
                                    size=10, random_state=1)
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

    def test_prior_lognormal(self):
        prior_lognormal = PriorLognormal(
            'a', ['loc_a', 'scale_a'], 'a_lognormal')
        # check the evaluation of the log-pdf
        prms = {'a': 2.0, 'loc_a': 1.0, 'scale_a': 1.0}
        self.assertEqual(
            stats.lognorm.logpdf(prms['a'], prms['loc_a'], prms['scale_a']),
            prior_lognormal(prms, 'logpdf'))
        # check the sampling-method (samples are checked one by one)
        prms = {'loc_a': 1.0, 'scale_a': 1.0}
        prior_samples = prior_lognormal.generate_samples(prms, 10, seed=1)
        sp_samples = stats.lognorm.rvs(1.0,  # this is scipy's shape parameter
                                       loc=prms['loc_a'], scale=prms['scale_a'],
                                       size=10, random_state=1)
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

    def test_prior_uniform(self):
        prior_uniform = PriorUniform('a', ['low_a', 'high_a'], 'a_uniform')
        # check the evaluation of the log-pdf
        prms = {'a': 0.5, 'low_a': 0.0, 'high_a': 1.0}
        self.assertEqual(
            stats.uniform.logpdf(prms['a'], prms['low_a'], prms['high_a']),
            prior_uniform(prms, 'logpdf'))
        # check the sampling-method (samples are checked one by one)
        prms = {'low_a': 0.0, 'high_a': 1.0}
        prior_samples = prior_uniform.generate_samples(prms, 10, seed=1)
        sp_samples = stats.uniform.rvs(
            loc=prms['low_a'], scale=prms['low_a'] + prms['high_a'],
            size=10, random_state=1)
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

    def test_prior_weibull(self):
        prior_weibull = PriorWeibull(
            'a', ['loc_a', 'scale_a', 'shape_a'], 'a_weibull')
        # check the evaluation of the log-pdf
        prms = {'a': 1.0, 'loc_a': 1.0, 'scale_a': 1.0, 'shape_a': 2.0}
        self.assertEqual(
            stats.weibull_min.logpdf(prms['a'], prms['shape_a'],
                                     prms['loc_a'], prms['scale_a']),
            prior_weibull(prms, 'logpdf'))
        # check the sampling-method (samples are checked one by one)
        prms = {'loc_a': 1.0, 'scale_a': 1.0, 'shape_a': 2.0}
        prior_samples = prior_weibull.generate_samples(prms, 10, seed=1)
        sp_samples = stats.weibull_min.rvs(
            prms['shape_a'], loc=prms['loc_a'], scale=prms['scale_a'],
            size=10, random_state=1)
        for s1, s2 in zip(prior_samples, sp_samples):
            self.assertEqual(s1, s2)

if __name__ == "__main__":
    unittest.main()
