# probeye changelog

## 1.0.2 (2021-Sep-22)
### Changed (2 item)
- Sensor class:
  * Added measurand- and unit-attribute.
- PositionSensor class
  * Renamed it to SensorWithCoordinates.
### Added (1 item)
- Changelog
  * Added CHANGELOG.md to repository.

## 1.0.1 (2021-Sep-20)
### Changed (1 item)
- PositionProperties class:
  * Its attributes (except for info and tex) have been made private. They cannot be changed directly from outside anymore. This should prevent the user from messing up a well-defined InferenceProblem.
  * The corresponding tests have been extendend to check this feature.

