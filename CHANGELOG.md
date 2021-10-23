# probeye changelog

## 1.0.X (2021-XX-XX)
### Changed
- revised arviz-based post-processing routine in create_posterior_plot, create_trace_plot

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
