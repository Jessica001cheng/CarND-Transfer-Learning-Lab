source activate carnd-term1
python feature_extraction.py --training_file resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet_cifar10_bottleneck_features_validation.p --batch_size 128 --epochs 10
