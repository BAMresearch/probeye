# probeye changelog

## 1.0.3 (2021-Sep-23)
### Changed
- InferenceProblem class
  * One can now add latent parameters without explicitly defining a prior. When not stated, the prior will be assumed to be uninformative. The shortest way to do so is by: problem.add_parameter('a', 'model') for example. This is aiming at maximum likelihood problems, where no priors need to be defined.
  * Removed independent '_priors'-attribute. It is now dynamically derived from the '_parameters'-attribute. By that, the priors are not stored at two locations anymore. The only place where a prior is stored now is the 'prior'-attribute of the corresponding parameter.
  * Added parameter-consistency checks in the method check_problem_consistency.
  * Removed private _add_prior-method. The adding of a prior is now done directly in the add_parameter method.
  * Removed add_prior-test in test_inference_problem.py.
- ParameterProperties class
  * Removed '_role'-attribute. There is still a 'role'-property, which is derived from the '_index'-attribute.
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
