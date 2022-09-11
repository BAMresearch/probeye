# probeye

[![Continuous integration](https://github.com/BAMresearch/probeye/actions/workflows/push.yaml/badge.svg)](https://github.com/BAMresearch/probeye/actions)
[![PyPI version](https://img.shields.io/pypi/v/probeye)](https://pypi.org/project/probeye/)
![python versions](https://img.shields.io/pypi/pyversions/probeye)
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/aklawonn/5eb707145cc7d75de25b43d25b13c972/raw/probeye_main_coverage.json)](https://en.wikipedia.org/wiki/Code_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package provides a transparent and easy-to-use framework for solving parameter estimation problems (i.e., [inverse problems](https://en.wikipedia.org/wiki/Inverse_problem)) chiefly via [sampling](https://ermongroup.github.io/cs228-notes/inference/sampling/) methods in a characteristic two-step approach. 

1. In the first step, the problem at hand is defined in a **solver-independent** fashion, i.e., without specifying which computational means are supposed to be utilized for finding a solution.
2. In the second step, the problem definition is handed over to a **user-selected solver**, that finds a solution to the problem. The currently supported solvers focus on Bayesian methods for posterior sampling.

Due to the broad variety of existing inference engine packages, _probeye_ does not contain self-written implementations of solvers but merely interfaces with existing ones. It currently provides interfaces with [emcee](https://emcee.readthedocs.io/en/stable/) for [MCMC sampling](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) and with [dynesty](https://dynesty.readthedocs.io/en/stable/) for [nested sampling](https://en.wikipedia.org/wiki/Nested_sampling_algorithm). It also provides two point-estimate solvers for [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) as  well as [maximum a-posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) estimates based on [scipy](https://scipy.org/).

The parameter estimation problems _probeye_ aims at are problems that are centered around forward models that are computationally expensive (e.g., parameterized finite element models), and the corresponding observations of which are not particularly numerous (typically around tens or hundreds of experiments). Such problems are often encountered in engineering problems where simulation models are calibrated based on laboratory tests, which are - due to their relatively high costs - not available in high numbers. 

The source code of _probeye_ is jointly developed by [_Bundesanstalt für Materialforschung und -prüfung (BAM)_](https://www.bam.de) and [_Netherlands Organisation for applied scientific research (TNO)_](https://www.tno.nl) for calibrating parameterized physics-based models and quantifying uncertainties in the obtained parameter estimates.

## Documentation
A documentation including explanations on the package's use as well as some examples can be found [here](https://probeye.readthedocs.io/en/latest/index.html).
