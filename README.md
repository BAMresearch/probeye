# probeye

[![Continuous integration](https://github.com/BAMresearch/probeye/actions/workflows/push.yaml/badge.svg)](https://github.com/BAMresearch/probeye/actions)
[![PyPI version](https://img.shields.io/pypi/v/probeye)](https://pypi.org/project/probeye/)
![python versions](https://img.shields.io/pypi/pyversions/probeye)
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/aklawonn/5eb707145cc7d75de25b43d25b13c972/raw/probeye_main_coverage.json)](https://en.wikipedia.org/wiki/Code_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package provides a transparent and easy-to-use framework for solving parameter estimation problems (i.e., [inverse problems](https://en.wikipedia.org/wiki/Inverse_problem)) in a characteristic two-step approach. 

1. In the first step, the problem at hand is defined in a **solver-independent** fashion, i.e., without specifying which computational means are supposed to be utilized for finding a solution.
2. In the second step, the problem definition is handed over to a **user-selected solver**, that finds a solution to the problem via frequentist methods, such as a [maximum likelihood fit](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), or Bayesian methods such as [Markov chain Monte Carlo sampling](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo).

The parameter estimation problems _probeye_ aims at are problems that are centered around forward models that are computationally expensive (e.g., parameterized finite element models), and the corresponding observations of which are not particularly numerous (around tens or hundreds of data points instead of thousands or millions). Such problems are often encountered in engineering problems where simulation models are calibrated based on laboratory tests, which are - due to their relatively high costs - not available in high numbers. 

The idea and source code of _probeye_ have been initially developed at the [_German Federal Institute for Materials Research and Testing (BAM)_](https://www.bam.de/Navigation/EN/About-us/Organisation/Organisation-Chart/President/Department-7/Division-77/division77.html) for calibrating parameterized constitutive material models and quantifying the uncertainties in the obtained estimates.

## Documentation
A documentation including explanations on the package's use as well as some examples can be found [here](https://probeye.readthedocs.io/en/latest/index.html).
