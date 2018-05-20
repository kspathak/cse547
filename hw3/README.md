# Homework 3

## Problem 4

Problem 4 involves training a model to categorize objects within bounding
boxes. The script [generate_examples.py](generate_examples.py) uses selective
search to propose regions that may contain an object and turns each region into
features with adaptive max pooling.
```
./docker_run.sh /hw3/generate_examples.py \
  --data_dir=data --dataset=training --size=small \
  --examples_output=data/patch_features_small/train2014.p
```
generates examples from the provided small features and outputs them into a
pickled file.

The script [run.py](run.py) trains a model that categorizes objects within a
region.
```
./docker_run.sh /hw3/run.py \
  --training_data=/data/patch_features_small/train2014.p \
  --validation_data=/data/patch_features_small/val2014.p \
  --model=multilayer_perceptron \
  --model_multilayer_perceptron_hidden_units=512 \
  --model_multilayer_perceptron_hidden_units=256 \
  --train_l2_regularization=4e-3 \
  --train_optimizer_learning_rate=5e-2 \
  --train_epochs=32 \
  --train_batch_size=16 \
  --train_summary_steps=1000 --train_evaluation_steps=4000 \
  --train_s3_output
```
trains a multi-layer perceptron model with two hidden layers and outputs that
model to S3.

Plots and tables that summarize the training run are generated in the
[problem4/training_log.ipynb](problem4/training_log.ipynb) notebook. Model
evaluation on the test set is done in
[problem4/model_evaluation.ipynb](problem4/model_evaluation.ipynb).