# Project

The project specification can be found at
[https://courses.cs.washington.edu/courses/cse547/18sp/project.html](https://courses.cs.washington.edu/courses/cse547/18sp/project.html). I
chose *Option 2*, object detection and retrieval with metric learning.

## Object Detection

Object detection first requires extracting features. This is done with the
[generate_examples.py](/hw3/generate_examples.py) script.
```
./docker_run.sh /hw3/generate_examples.py \
  --examples_output=/data/patch_features_small/train2014_positive.p \
  --data_dir=/data --size=small \
  --bbox_file_path=/data/bboxes/train2014_bboxes.p --dataset=train
```
This needs to be done for the `validation` and `test` datasets, too.

The multi-layer perceptron can be trained with the [run.py](run.py) script.
```
./docker_run.sh /project/run.py \
  --model=multilayer_perceptron --train_l2_regularization=2.5e-3 \
  --train_epochs=32 --train_optimizer_learning_rate=3e-2 \
  --train_s3_output
  --training_data=/data/patch_features_small/train2014_positive_001.p,/data/patch_features_small/train2014_positive_002.p,/data/patch_features_small/train2014_positive_003.p,/data/patch_features_small/train2014_positive_004.p \
  --validation_data=/data/patch_features_small/val2014_positive.p \
  --train_batch_size=128 --train_summary_steps=1000 --train_evaluation_steps=4000 \
  --model_multilayer_perceptron_hidden_units=1024 \
  --model_multilayer_perceptron_hidden_units=256 \
  --model_multilayer_perceptron_dropout=0.5 \
  --train_optimizer_sgd_nesterov --train_optimizer_sgd_momentum=0.9
```

All these scripts required a machine with at least 32 GB of RAM.

Training progress is plotted in
[training_log.ipynb](object_detection/training_log.ipynb). The model is
evaluated against the test set in
[model_evaluation.ipynb](object_detection/model_evaluation.ipynb).

## Metric Learning

Neural code metric learning was tried, which means extracting the last hidden
layer of a neural network. Given a model, [embed.py](embed.py) extracts the last
layer and writes the output to disk.
```
./docker_run.sh /project/embed.py \
  --s3_model_key=project/train/model_103740db-7181-4771-aa0b-cfa7cc407cf8.pkl \
  --s3_training_log_key=project/train/training_log_103740db-7181-4771-aa0b-cfa7cc407cf8.pkl \
  --data=data/patch_features_small/train2014_positive_001.p,data/patch_features_small/train2014_positive_002.p,data/patch_features_small/train2014_positive_003.p,data/patch_features_small/train2014_positive_004.p \
  --output=data/patch_features_small/train2014_embeddings.p
```
This is also done for the `validation` and `test` datasets.

Then, [metric_learning.ipynb](metric_learning/metric_learning.ipynb) uses this
distance function to create a ball tree and compute the nearest neighbors for
each example. These neighbors are cached and written to disk.

Using the cached neighbors, the quality of this metric is evaluated in
[metric_learning_evaluation.ipynb](metric_learning_evaluation.ipynb).