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

A [Docker](https://hub.docker.com/) container is provided to make running the code easier. It can be built with the following command.
```sh
docker build -t cse547 .
```
Note the period at the end. My [Dockerfile](Dockerfile) is based off the one provided by [floydhub/dockerfiles](https://github.com/floydhub/dockerfiles/blob/master/dl/pytorch/0.3.1/Dockerfile-py3).

Running the script `./docker`

## Development



## Running on Amazon Web Services (AWS)

TODO(ppham27)

## Authors

* **Philip Pham** - [ppham27](https://github.com/ppham27)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
