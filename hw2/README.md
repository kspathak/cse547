# Homework 2

The full assignment can be found in [hw2.pdf](hw2.pdf). Solutions to all the
problems are described in detail in the [writeup.pdf](writeup.pdf).

## Problem 4

The multi-label classification models can be trained with the [run.py](run.py)
script. The instructions and script are very similar to the [Homework 1 Problem
6](/hw1/README.md#problem-6). This script requires the `features2` datasets. See
[/data/README.md](/data/README.md) for details.

The easiest way to run the script is with a Docker container: `./docker_run.sh
/hw2/run.py`. Here are some examples of configurations that you might want to
run.

Linear model with cross-entropy loss and SGD:
```
./docker_run.sh /hw2/run.py --model=linear \
  --train_optimizer=sgd --train_optimizer_learning_rate=5e-4 \
  --train_optimizer_nesterov --train_optimizer_momentum=0.9 \
  --train_loss_function=cross_entropy --train_l2_regularization=0.5 \
  --train_epochs=8
```

Multi-layer perceptron with Adagrad:
```
./docker_run.sh /hw2/run.py --model=multilayer_perceptron \
  --train_optimizer=adagrad --train_optimizer_learning_rate=1e-4 \
  --train_loss_function=cross_entropy --train_l2_regularization=1e-2 \
  --train_epochs=16
```

The code for plots can be found in the [Problem 4](problem4/Problem%204.ipynb)
notebook.
