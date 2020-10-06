
# Developing
Before you start, please install all required packages:

    pip install --upgrade -r dev/requirements-dev.txt

## Testing
Before commiting, please test the package:

    python -m pytest
    flake8

## Releasing a new version on PyPI
Update version: 
* The `setup(version=...)` parameter in the setup.py file
* The package’s `__version__` attribute, in Python
* Documentation 

Publish on PyPI:

    python setup.py sdist
    python setup.py bdist_wheel
    twine upload dist/*
