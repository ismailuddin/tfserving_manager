# TensorFlow Serving Manager
> A simple easy to use API for interacting with the TensorFlow Serving server
> to make predictions and load configuration files

## Installation
First clone the repository, and then install using the command:

```shell
$   pip install .
```

## Usage
Refer to the notebooks for examples of how to use the package.

Functions are provided to interact with both the RESTful API endpoints, as well as the gRPC API endpoints.

## Documentation
Documentation can be built using the `Makefile`. First, run `make html` inside the `docs/` directory. Then, launch a server inside `docs/build/html` using the command `python -m http.server`.