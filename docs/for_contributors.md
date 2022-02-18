(sec:for_contributors)=
# For contributors

All contributions are welcome, feel free to create a pull
request on [GitHub][github_repository]. Please note that we are using [black](https://pypi.org/project/black/) as a format-checker and [mypy](http://mypy-lang.org/) as a static type-checker to pre-check any pushes to a pull-request. If your code is not in line black/mypy, no unit tests will be run.

[github_repository]: https://github.com/BAMresearch/probeye

## Documentation

We use [sphinx](https://www.sphinx-doc.org/en/master/) & [ReadTheDocs](https://readthedocs.org/) for documenting.

Test your changes locally:
* Install the documentation dependencies (`setup.cfg`: `docs`).
* Local build: from the `docs` directory execute `make html` (on Windows: `.\make.bat html`).

## Testing

We use `pytest` for testing.