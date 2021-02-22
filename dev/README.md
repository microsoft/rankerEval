
# Developing
Before you start, please install all required packages:

    pip install --upgrade -r dev/requirements-dev.txt

You can install the package in editable mode to see updates immediately:

    pip install -e .

## Testing
Before commiting, please test the package:

    python -m pytest
    flake8

### Fixing code styles
To fix most of lint errors, run autopep8

    autopep8 --in-place -r .

You can preview changes via

    autopep8 --diff -r .

## Releasing a new version on PyPI
Update version: 
* The `setup(version=...)` parameter in the setup.py file
* The package’s `__version__` attribute, in Python
* Documentation: conf.py

Publish on PyPI:

    python setup.py sdist
    python setup.py bdist_wheel
    twine upload dist/*
