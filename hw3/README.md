# Homework 3

## Problem 4

Problem 4 involves training a model to categorize objects within bounding
boxes. The script [generate_examples.py](generate_examples.py) uses selective
search to propose regions that may contain an object and turns each region into
features with adaptive max pooling.

The script [run.py](run.py) trains a model that categorizes objects within a
region.

