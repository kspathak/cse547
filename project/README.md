```
/project/run.py --model=multilayer_perceptron --train_epochs=32 --train_l2_regularization=4e-3 --train_optimizer_learning_rate=5e-2 --train_s3_output --training_data=/data/patch_features_small/train2014_positive_001.p,/data/patch_features_small/train2014_positive_002.p,/data/patch_features_small/train2014_positive_003.p,/data/patch_features_small/train2014_positive_004.p --validation_data=/data/patch_features_small/val2014_positive.p --train_batch_size=16 --train_summary_steps=1000 --train_evaluation_steps=4000 --model_multilayer_perceptron_hidden_units=512 --model_multilayer_perceptron_hidden_units=256 --train_optimizer_sgd_nesterov --train_optimizer_sgd_momentum=0.9
```

```
/project/run.py --model=multilayer_perceptron --train_epochs=32 --train_l2_regularization=4e-3 --train_optimizer_learning_rate=1e-1 --train_s3_output --training_data=/data/patch_features_tiny/train2014_positive.p --validation_data=/data/patch_features_tiny/val2014_positive.p --train_batch_size=32 --train_summary_steps=1000 --train_evaluation_steps=4000 --model_multilayer_perceptron_hidden_units=512 --model_multilayer_perceptron_hidden_units=256 --train_optimizer_sgd_nesterov --train_optimizer_sgd_momentum=0.9
```

```
./docker_run.sh python /project/embed.py \
  --s3_model_key=project/train/model_efe2a107-3fb7-49bf-a15d-abf58949ae87.pkl \
  --s3_training_log_key=project/train/training_log_efe2a107-3fb7-49bf-a15d-abf58949ae87.pkl \
  --data=data/patch_features_small/train2014_positive_001.p,data/patch_features_small/train2014_positive_002.p,data/patch_features_small/train2014_positive_003.p,data/patch_features_small/train2014_positive_004.p \
  --output=data/patch_features_small/train2014_embeddings.p
```