# CSE 547

Code to solve exercises for UW's [CSE 547](https://courses.cs.washington.edu/courses/cse547/18sp/).

## Data

To train models, you need data. You can find the data needed in the [CSE 547 Team Drive](https://drive.google.com/drive/folders/1xdtYAOvOxwPMVydrsxNAA30YIOP6RoEJ). You may need to request access.

Download the archives and put the output in a `data` folder. Assuming you cloned this repository and didn't change the name, your directory structure should look like:

```
cse547/data/
├── annotations
│   ├── instances_test2014.json
│   ├── instances_train2014.json
│   └── instances_val2014.json
├── features_small
│   ├── test2014.p
│   ├── train2014.p
│   └── val2014.p
├── features_tiny
│   ├── test2014.p
│   ├── train2014.p
│   └── val2014.p
└── train2014
```

## Running the Code

A [Docker](https://hub.docker.com/) container is provided to make running the code easier. A Docker image can be built with the following command.
```sh
docker build -t cse547 .
```
Note the period at the end. My [Dockerfile](Dockerfile) is based off the one provided by [floydhub/dockerfiles](https://github.com/floydhub/dockerfiles/blob/master/dl/pytorch/0.3.1/Dockerfile-py3).

Running the script `./docker_run.sh` will start a Docker container that runs a [Jupyter](http://jupyter.org/) notebook server. Follow the instructions to connect. Your current working directory (probably the root of this repository) will be mounted into the `/local` folder. You can browse to an Jupyter Notebook file (`*.ipynb`) and try it.

You can run other commands inside the container as well. Two particularly useful commands for debugging are:

- `./docker_run.sh python` to open a Python interpreter; and
- `./docker_run.sh bash` to start a terminal session.

### Homeworks

See the `README.md` files in the corresponding homework directories on how to run that code.

- [Homework 1](hw1/README.md)

### Troubleshooting

Your Docker container may run out of memory. You can increase the memory through the UI for the Docker daemon.

## Development

The Docker container makes it easy to run code, but writing code is best done in a Jupyter Notebook or on your local machine. Local changes will be reflected inside the `/local` folder inside the container, so you can quickly test your code.

The core library code lives inside the [cse547](cse547) directory. In order these changes to reflected without constantly needing to rebuild new Docker images, you can run `pip install -e .` inside the `/local` directory inside of the Docker container. If testing code in a Jupyter Notebook, it may be useful to use the `reload` function of [importlib](https://docs.python.org/3/library/importlib.html).

## Running on Amazon Web Services (AWS)

TODO(ppham27): Write later

## Authors

* **Philip Pham** - [ppham27](https://github.com/ppham27)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
