# Homework 1

The full assignment can be found in [hw1.pdf](hw1.pdf). Solutions to all the
problems are described in detail in the [writeup.pdf](writeup.pdf).

A couple of the problems required code. Instructions on how to run that code are
below.

## Problem 4

This code lives in the [hw1/problem4](/hw1/problem4) directory. It's a
self-contained Jupyter notebook that only depends on
[PyTorch](https://github.com/pytorch/pytorch),
[Matplotlib](https://github.com/matplotlib/matplotlib), and
[NumPy](https://github.com/numpy/numpy).

## Problem 6

[run.py](run.py) uses library code from the [cse547 module](/cse547), so it is
easiest to run inside a Docker container. All commands should be run from the
root of this repository. Remember to populate the [data](/data) directory.

Here are some examples.

### Training the Linear Model with Tiny Features

By default it will training on the tiny features with the linear model.

```sh
./docker_run.sh /hw1/run.py
```

### Training the Linear Model with Small Features

See [run.py](run.py) or run `./docker_run.sh /hw1/run.py --help` for a full list of flags.

```sh
./docker_run.sh /hw1/run.py \
  --size=small \
  --train_batch_size=16 \
  --train_l2_regularization=6e-4 \
  --train_optimizer_learning_rate=1e-6
```

### Training the Multi-Layer Perceptron

This model takes considerably longer to train and has an additional hyperparameter.

```sh
./docker_run.sh /hw1/run.py \
  --size=small \
  --model=multilayer_perceptron \
  --model_multilayer_perceptron_hidden_units=150 \
  --train_epochs=64
```

### Running on AWS

You can train the models as a batch job following the instructions
[here](https://github.com/ppham27/cse547/blob/master/README.md#running-on-amazon-web-services-aws). I
needed a machine with 4 CPUs and 8GB of memory for the job to complete. The
relevant command is `/hw1/run.py`.

### Reproducing the Graphs

The last line of the AWS Batch logs has the loss and accuracy. Copy and paste
this data into the [Problem 6](problem6/problem6.ipynb) notebook, and run the
code.

### Troubleshooting

If your Python process is killed, it's likely due to an out-of-memory (OOM) error see [Troubleshooting](/README.md/#troubleshooting).
