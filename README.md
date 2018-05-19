# CSE 547

Code to solve exercises for UW's [CSE
547](https://courses.cs.washington.edu/courses/cse547/18sp/). The code lives at
[https://gitlab.cs.washington.edu/pmp10/cse547](https://gitlab.cs.washington.edu/pmp10/cse547).

## Data

To train models, you need data. See [data/README.md](data/README.md) on
instructions on how to populate the `data` folder.

## Running the Code

A [Docker](https://hub.docker.com/) container is provided to make running the
code easier. A Docker image can be built with the following command.  ```sh
docker build -t cse547 .  ``` Note the period at the end. My
[Dockerfile](Dockerfile) is based off the one provided by
[floydhub/dockerfiles](https://github.com/floydhub/dockerfiles/blob/master/dl/pytorch/0.3.1/Dockerfile-py3).

Running the script `./docker_run.sh` will start a Docker container that runs a
[Jupyter](http://jupyter.org/) notebook server. Follow the instructions to
connect. Your current working directory (probably the root of this repository)
will be mounted into the `/local` folder. You can browse to an Jupyter Notebook
file (`*.ipynb`) and try it.

You can run other commands inside the container as well. Two particularly useful commands for debugging are:

- `./docker_run.sh python` to open a Python interpreter; and
- `./docker_run.sh bash` to start a terminal session.

### Homeworks

See the `README.md` files in the corresponding homework directories on how to run that code.

- [Homework 1](hw1/README.md)
- [Homework 2](hw2/README.md)
- [Homework 3](hw3/README.md)

### Troubleshooting

Your Docker container may run out of memory. You can increase the memory through
the UI for the Docker daemon.

## Development

The Docker container makes it easy to run code, but writing code is best done in
a Jupyter Notebook or on your local machine. Local changes will be reflected
inside the `/local` folder inside the container, so you can quickly test your
code.

The core library code lives inside the [cse547](cse547) directory. In order for
these changes to reflected without constantly needing to rebuild new Docker
images, you can run `pip install -e .` inside the `/local` directory inside of
the Docker container. If testing code in a Jupyter Notebook, it may be useful to
use the `reload` function of
[importlib](https://docs.python.org/3/library/importlib.html).

## Running on Amazon Web Services (AWS)

### AWS Batch

Most of the training jobs are best run as batch jobs. AWS Batch allows you to
submit jobs by specifying an EC2 AMI, Docker container, and a command.

#### Configuring Your EC2 AMI

You need to make the data available to your EC2 instance. One way to do this is
put the data in EFS and then mount the volume.

To get the data into EFS, you can put it in S3 first and then copy it to
EFS. Assuming your data is in EFS, launch an [Amazon ECS-Optimized
AMI](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-optimized_AMI.html).

On this instance, install the
[efs-mount-helper](https://docs.aws.amazon.com/efs/latest/ug/using-amazon-efs-utils.html#efs-mount-helper)
and configure your instance to [automatically mount your EFS file system on
reboot](https://docs.aws.amazon.com/efs/latest/ug/mount-fs-auto-mount-onreboot.html). Create
a new AMI from this instance. Feel free to stop it now.

#### Pushing Your Docker Container to ECR

From the [Elastic Container Service (ECS)](https://aws.amazon.com/ecs/), create
a new repository. Follow the commands to push your container to this repository.

#### Setting Up the Batch Job

Create a compute environment that uses the AMI you just created. Create a job
queue that submits to this compute environment.

In your job definition, specify the container you just pushed to ECR. In the
**Volumes** section, create a reference to the directory where you mounted your
EFS file system. In the **Mount points** section, mount this volume to `/data`.

Now specify the command that you'd like to run from the Docker container's root
along any flags. Allocate the necessary resources. Submit the job. You can
change flag values or the command without creating a new job definition revision
when searching for hyperparameters.

When the job is running, you'll see logs in
[CloudWatch](https://aws.amazon.com/cloudwatch/) by following the link after
clicking on the job.

## Authors

* **Philip Pham** - [ppham27](https://github.com/ppham27)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
