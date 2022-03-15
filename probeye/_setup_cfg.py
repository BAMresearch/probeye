"""This module contains functions that remove version number constraints from
selected packages in the setup.cfg file. It is meant to be used during CI: to test if
the package works with the latest versions of its dependencies."""

import configparser
import re
import os
from typing import List


def version_constraint_free_packages(setup_cfg_packages: str) -> List[str]:
    """
    Remove version constraints from the packages listed in `options_field` (provided
    as input argument or read from `setup.cfg`).

    Parameters
    ----------
    setup_cfg_packages: str
        Packages with optional version constraints as parsed from a syntactically
        correct `setup.cfg` file.

    Returns
    -------
    List[str]
        `install_requires` packages without version constraints.
    """
    # to be able to match the last package in all cases
    setup_cfg_packages = "\n" + setup_cfg_packages + "\n"

    # get plain (version constraint free) package names, they are listed using a
    # "list-semi": dangling list or string of semicolon-separated values
    # https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#options
    packages_without_version = re.findall(
        r"(?<=(?:\n|;)).*?(?=<|=|>|!|;|\n|;])", setup_cfg_packages
    )
    # strip possible remaining spaces and drop empty string elements
    packages_without_version = [s.strip() for s in packages_without_version]
    packages_without_version = list(filter(None, packages_without_version))

    return packages_without_version


def version_constraint_free_dependencies(
    options_field: str, test: bool = False, setup_cfg: str = "setup.cfg"
) -> None:
    """
    Remove version constraints from the packages listed in `options_field` (
    provided as input argument or read from `setup.cfg`) and overwrite the
    `setup.cfg` file with the version constraint-free dependencies. Useful in a CI
    pipeline to test if the package works with the latest versions of its dependencies.

    Parameters
    ----------
    options_field: str
        Field name in `setup.cfg` options that list package dependencies,
        e.g. `"install_requires"`. For further information see:
        https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#options
    test
        Flag only used for testing. If this method is tested during CI, it is set to
        True. In all other cases it should be False.
    setup_cfg
        Path to the setup.cfg file. Added as an argument for testing.

    """
    config = configparser.ConfigParser()
    config.read(setup_cfg)
    packages = config["options"][options_field]

    packages_without_version = version_constraint_free_packages(packages)
    setup_cfg_packages_without_version = "\n" + "\n".join(packages_without_version)

    config["options"][options_field] = setup_cfg_packages_without_version

    # just for testing during CI
    if test:
        setup_cfg = "setup_test.cfg"

    with open(setup_cfg, "w") as configfile:
        config.write(configfile)

    # just for testing during CI
    if test:
        os.remove(setup_cfg)


# to ease running the version number constraint removal from the command line
if __name__ == "__main__":
    # this is meant to be run only during CI: to test if the package works with the
    # latest versions of its dependencies
    version_constraint_free_dependencies(
        options_field="install_requires"
    )  # pragma: no cover
