(sec:installation)=
# Installation
The source code is entirely written in Python and can be installed either from PyPI or directly from the latest version on GitHub. Compatible Python versions (at least tested ones) are 3.6 up to 3.9.

## pip (PyPI)

Since [_probeye_](https://pypi.org/project/probeye/) is frequently updated on the Python package index ([PyPI](https://pypi.org/)), you can install the most recent stable version simply by running _pip_ from the command line:

```bash
pip install probeye
```

Please note that there are several dependencies which might take some time to download and install, if they are not installed in your environment yet.

## pip (GitHub)

If you want to install the package directly from the latest version on [GitHub][repository] (for example when this version is not published on PyPI yet), just clone the [repository][repository] and install the package locally by running from the
root of the repository the following command:

````{tab} User
```bash
pip install .
```
````

````{tab} Developer (Windows)
```bash
pip install -e .[tests,lint_type_checks,docs]
```
````

````{tab} Developer (Linux)
```bash
pip install --user ".[tests,lint_type_checks,docs]"
```
````

Please note that there are several dependencies which might take some time to download and install, if they are not installed in your environment yet.


[repository]: https://github.com/BAMresearch/probeye
