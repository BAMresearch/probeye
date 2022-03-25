# probeye

This package provides a transparent and easy-to-use framework for solving parameter estimation problems (i.e., [inverse problems](https://en.wikipedia.org/wiki/Inverse_problem)) in a characteristic two-step approach. 

1. In the first step, the problem at hand is defined in a **solver-independent** fashion, i.e., without specifying which computational means are supposed to be utilized for finding a solution.
2. In the second step, the problem definition is handed over to a **user-selected solver**, that finds a solution to the problem via frequentist methods, such as a [maximum likelihood fit](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), or Bayesian methods such as [Markov chain Monte Carlo sampling](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo).

The parameter estimation problems _probeye_ aims at are problems that are centered around forward models that are computationally expensive (e.g., parameterized finite element models), and the corresponding observations of which are not particularly numerous (typically around tens or hundreds of experiments). Such problems are often encountered in engineering problems where simulation models are calibrated based on laboratory tests, which are - due to their relatively high costs - not available in high numbers. 

The source code of _probeye_ is jointly developed by [_Bundesanstalt für Materialforschung und -prüfung (BAM)_](https://www.bam.de) and [_Netherlands Organisation for applied scientific research (TNO)_](https://www.tno.nl) for calibrating parameterized physics-based models and quantifying uncertainties in the obtained parameter estimates.




```{toctree}
---
hidden:
---

installation
auto_examples/index
motivation
components
api
```

```{toctree}
---
hidden:
maxdepth: 2
caption: Development
---

for_contributors
```
