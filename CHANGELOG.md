# probeye changelog

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
