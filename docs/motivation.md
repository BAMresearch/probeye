(sec:motivation)=

# Motivation

The forward model is a python wrapper to a potentially complex simulation software. The
input is usually a set of scalar or vector valued parameters, and the output is a set of
scalars, vectors or matrix type outputs. Usual interfaces such as `scipy.optimize`
only consider a single input and output vector. However, this makes it very difficult in
the optimization or inference procedure to extract relevant information.

For the input, specifying one specific index in the global parameter vector to each
latent variable is tricky.

* What exactly is supposed to be a model input is often difficult to define a priori
  when starting the inference problem, since in many cases during the progress of the
  project we experienced the problem that additional parameters have to be added or the
  parameters change their type from being deterministic to be inferred.
* Combinations of different data sets with different model parameters are difficult to
  handle. If another model with additional parameters is added, this changes all
  subsequent entries.
* Parameters can be either shared between experiments or individual per data set. Assume
  you have 10 tensile tests that are influenced by temperature - but you forgot to
  measure this. The single material parameter Young's modulus *E* is shared between all
  experiments, the temperature *T* for each individual experiment might be another
  parameter to be identified for each individual experiment. This results in two
  parameters per individual forward model (E*, *T*), but on the global inference level
  there are 11 parameters (Young's modulus plus 10 temperatures).

For the output, just taking the norm of a residual vector is often misleading.

* The accuracy of different outputs is often not constant. This depends e.g. on the
  units used, but if your output contains sensors that e.g. measure temperature,
  displacement and relative humidity, the error of each sensor has to be weighted
  differently. For a complete Bayesian approach, it might be advantageous to define
  individual noise terms for each sensor type that is either identified as a
  hyper-parameter or defined by the user based on the accuracy of the measurement
  device.
* The sensor output is often correlated in space and/or time. As a consequence, a
  covariance matrix has to be specified that defines a new metric for weighting the
  error. This should not be performed within the forward model, but within the inference
  task. As a consequence, it is necessary to extract relevant information such as the
  sensor coordinates and time stamps related to each individual forward model output.
* Combining data from different experiments is difficult. One problem might be to
  identify material parameters (constitutive parameters) for a given material with a
  dataset comprising 10 different simple tension tests as well as 5 additional
  compression tests and another 3-point bending test. This requires three different
  forward models (simple tension, compression, 3-point bending) with a flexible amount
  of data sets (10, 5, 1) and different outputs. It is essential to distinguish that in
  the inference procedure.