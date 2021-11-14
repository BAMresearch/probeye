(sec:installation)=
# Installation

## pip

You can install the most recent stable version by using:

```bash
pip install probeye
```

## From the source

Clone the [repository][repository] and install the package locally using (run from the
root of the repository):

````{tab} User
```bash
pip install .
```
````

````{tab} Developer
```bash
pip install -e .[tests,lint_type_checks,docs]
```
````

[repository]: https://github.com/BAMresearch/probeye
