# probeye changelog

## 3.0.0 (2022-Jun-14)
### Changed
- The definition of the latent parameter's priors was revised (from tuples to objects).
- The forward model's response method is stripped from the problem definition so that the pure definition of an inverse problem indeed does not contain any computing methods.
- After parameters one must now define experiments and then the forward models.
- The four main attributes of InverseProblem are not private anymore to allow for manipulations in the solver.
- The solver routines were revised.
- Fixed some problems with the plotting routines.
### Added
- Distribution classes have been added to make the prior-definition more clear.
- A new experiment class was added.
- A new correlation_model class was added.

## 2.3.2 (2022-May-30)
### Added
- The Scipy-solver was equipped with a maximum a-posteriori estimation method.

## 2.3.1 (2022-May-30)
### Removed
- The 'knowledge_graph_import.py' script was removed. The functionality was moved to the integration test 'test_query_results.py'.

## 2.3.0 (2022-May-16)
### Added
- Added new submodule 'ontology' for an import/export of the problem's knowledge graph
- Added an ontology file 'parameter_estimation_ontology.owl'

## 2.2.0 (2022-Apr-26)
### Changed
- The forward model's 'definition'-method was renamed to 'interface'.
- The user-definition of the forward model's 'interface'-method is now mandatory.
- The prior-parameters of the normal and lognormal distribution have been renamed from 'loc', 'scale' to 'mean', 'std'.
- The pyro-solver has (and its dependencies have) been removed from the package. All solvers are now based on numpy-arrays.
- Fixed a bug resulting from an update of numpy causing an error when using dynesty with a given seed.
- Removed method InverseProblem.transform_experimental_data since it is no longer necessary without the pyro/torch-solver.
- The specification of a parameter's type (model, prior or likelihood) is no longer necessary (it can be detected automatically).
- The multivariate normal prior is now an own prior type and not anymore included in the normal prior class.
- Updated the docs (however, still work to be done).

## 2.1.5 (2022-Mar-31)
### Changed
- The main class InferenceProblem was renamed to InverseProblem.

## 2.1.4 (2022-Mar-27)
### Changed
- Before the evaluation of a likelihood function, the parameter domains are checked.
### Added
- Latent parameters have domains now. A respective class has been added to the parameters.py-file.
- More tests have been added.

## 2.1.3 (2022-Mar-24)
### Changed
- extended the documentation

## 2.1.2 (2022-Mar-16)
### Changed
- Fixed a bug in the pyro-solver.
- Extended the linear_regression test to two and one variables (special cases).

## 2.1.1 (2022-Mar-14)
### Changed
- Optimized general log-likelihood evaluation.
- Faster method for scaling covariance matrix by model output.

## 2.1.0 (2022-Feb-18)
### Changed
- extended the documentation in README.md
### Added
- added a 'readthedocs' documentation

## 2.0.3 (2022-Feb-16)
### Changed
- For simple cases, it is now not required anymore to specify the likelihood model's experiments and sensors when adding them to an inference problem.
- Removed the underscores in the inference submodules scipy_, emcee_, torch_ and dynesty_. These underscores resulted in some confusion ("Are functions from these modules intended to be used by the user?").

## 2.0.2 (2022-Feb-15)
### Added
- Added some error checks after calling the forward model's definition method to prevent wrong use and weird error messages.

## 2.0.1 (2022-Feb-14)
### Added
- Added a 'definition' method to the forward model that allows a clearer way of defining a forward model's parameters, input and output sensors. However, the old way of defining a forward model's interface still works too.

## 2.0.0 (2022-Feb-09)
### Changed
- revised the entire noise model framework to account for correlation setups
- the 'noise_model' is now generally relabeled as 'likelihood_model' since the term 'noise_model' resulted in some confusion
- when correlation effects are modeled, the log-likelihood function is now much more efficiently evaluated using the tripy package developed by Ioannis Koune (TNO)

## 1.0.17 (2022-Jan-19)
### Changed
- Noise models must now be added after all experiments have been added. This means that the relevant experiments can be assigned to the noise models right after they are added (this happens automatically). This in turn makes it unnecessary to copy a problem and assign the experiments as a kind of post-processing step in the different solvers.

## 1.0.16 (2021-Dec-03)
### Added
- added truncated normal distribution to the scipy-based prior classes

## 1.0.15 (2021-Dec-01)
### Added
- added dynesty-solver (probeye/inference/dynesty_ submodule)

## 1.0.14 (2021-Nov-29)
### Changed
- revised arviz-based post-processing routines

## 1.0.13 (2021-Nov-29)
### Changed
- fixed a bug with the probeye.txt file, which was missing in the build

## 1.0.12 (2021-Nov-28)
### Changed
- modified code format according to standard black and introduced type hints

## 1.0.11 (2021-Nov-10)
### Added
- added summary attribute to solver classes

## 1.0.10 (2021-Nov-10)
### Changed
- the evaluate_model_response method was moved from InferenceProblem to the solver classes
### Added
- added integration test for maximum likelihood estimation with uninformative priors

## 1.0.9 (2021-Nov-09)
### Changed
- deep-copying the inference problem in the solver classes is wrapped with try/except, since not all problems can be deepcopied

## 1.0.8 (2021-Nov-09)
### Added
- added feature of vector-valued parameters
- added new ParameterProperties attribute 'dim' for specifying a parameter's dimension
- added explicitly defined Jacobian in test_linear_regression to show how a user can define the Jacobian to his forward model
- added new integration test 'test_multivariate_prior.py' to show use of vector-valued parameters
### Changed
- changed PriorNormal to cover multivariate distributions
- moved 'probeye.txt' into the probeye directory
- the probeye_header function does not read from setup.cfg anymore (this file is not available after installing the package)
- more plots with different options are now generated in the integration tests

## 1.0.7 (2021-Oct-29)
### Added
- added probeye header-file 'probeye.txt'
- added loguru dependency
### Changed
- changed various print-commands to logging-commands using loguru

## 1.0.6 (2021-Oct-26)
### Added
- Added ScipySolver-class which provides a maximum likelihood solver
### Changed
- restructured solver routines for emcee and pyro to methods of corresponding solver-classes

## 1.0.5 (2021-Oct-22)
### Changed
- computation of Jacobian reshaped from 3d to 2d array
- corresponding modifications in pyro-solver
- adopted corresponding tests

## 1.0.4 (2021-Oct-20)
### Added
- torch submodule (in inference submodule)
- arviz-based post-processing for sampling methods (in postprocessing submodule)
- integration test with a correlation example
- new subroutines: flatten_generator, flatten, process_spatial_coordinates, translate_prms_def
- new script: correlation_models.py in inference/emcee module
- general plot-method in PriorBase
### Changed
- tool-independent noise model (in definition submodule)
- tool-independent normal noise model (in definition submodule)
- normal noise model for emcee (revised to be able to account for correlation)

## 1.0.3 (2021-Oct-04)
### Changed
- InferenceProblem class
  * One can now add latent parameters without explicitly defining a prior. When not stated, the prior will be assumed to be uninformative. The shortest way to do so is by: problem.add_parameter('a', 'model') for example. This is aiming at maximum likelihood problems, where no priors need to be defined.
  * Removed independent '_priors'-attribute. It is now dynamically derived from the '_parameters'-attribute. By that, the priors are not stored at two locations anymore. The only place where a prior is stored now is the 'prior'-attribute of the corresponding parameter.
  * Added parameter-consistency checks in the method check_problem_consistency.
  * Removed private _add_prior-method. The adding of a prior is now done directly in the add_parameter method.
  * Removed add_prior-test in test_inference_problem.py.
  * Moved most code of add_parameter-method to Parameters-class.
- ParameterProperties class
  * Removed '_role'-attribute. There is still a 'role'-property, which is derived from the '_index'-attribute.
  * Added 'changed_copy'-method for writing on the private attributes.
  * Added method check_consistency.
  * Added corresponding tests in test_parameter.py.

## 1.0.2 (2021-Sep-22)
### Changed (1 item)
- Sensor class:
  * Combined with PositionSensor. Now, the basic Sensor class provides the option to define sensor coordinates.
### Removed (1 item)
- PositionSensor class
  * Removal due to merge with the basic Sensor class.
### Added (1 item)
- Changelog
  * Added CHANGELOG.md to repository.

## 1.0.1 (2021-Sep-20)
### Changed (1 item)
- PositionProperties class:
  * Its attributes (except for info and tex) have been made private. They cannot be changed directly from outside anymore. This should prevent the user from messing up a well-defined InferenceProblem.
  * The corresponding tests have been extended to check this feature.