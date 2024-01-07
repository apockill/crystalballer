# crystalballer
Combining projection mapping and head tracking to make awesome mystical interactive displays!
_________________

[![PyPI version](https://badge.fury.io/py/crystalballer.svg)](http://badge.fury.io/py/crystalballer)
[![Test Status](https://github.com/apockill/crystalballer/workflows/Test/badge.svg?branch=main)](https://github.com/apockill/crystalballer/actions?query=workflow%3ATest)
[![Lint Status](https://github.com/apockill/crystalballer/workflows/Lint/badge.svg?branch=main)](https://github.com/apockill/crystalballer/actions?query=workflow%3ALint)
[![codecov](https://codecov.io/gh/apockill/crystalballer/branch/main/graph/badge.svg)](https://codecov.io/gh/apockill/crystalballer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
_________________

[Read Latest Documentation](https://apockill.github.io/crystalballer/) - [Browse GitHub Code Repository](https://github.com/apockill/crystalballer/)
_________________

## Running the Project

### Setting up the Software
To run this project, simply have docker installed, and run the following command, where 
you can replace COMMAND with the script you'd like to run in the docker environment:

```shell
./run COMMAND
```

This will build, run, and execute the container with the main project script. It will 
also enable X11 Forwarding, so that visuals can pass through the docker container.

Usable scripts can be found in the `scripts` section of the `pyproject.toml`.

### Setting up the Hardware

1. Connect the Gakken display to the computer via HDMI, and set the resolution to 800x600.

2. Connect the Oak-D Lite to the computer via USB.

### Validating GPU works
You can check if your docker installation works with GPU by seeing if nvidia-smi works
correctly. You can do this by running the following command:
```shell
./run nvidia-smi
```

## Development

### Installing python dependencies
```shell
poetry install
```

### Running Tests
```shell
pytest .
```

### Formatting Code
```shell
bash .github/format.sh
```

### Linting
```shell
bash .github/check_lint.sh
```