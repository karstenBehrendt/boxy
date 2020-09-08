# Simple 2D bounding box detection with tensorflow object detection

1. to_tfrecords.py creates the dataset based on boxy labels and images
2. quick_visualize_tfrecords.py shows samples from the dataset with boxes
3. https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html shows how to install and train tensorflow object detection. You will have to download some pre-trained models and slightly adapt the config to train them. For example, if you have an 8GB or smaller GPU, you will likely have to adapt the training batch size
4. The training command should be something like: "python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn". Now we wait.

You pretty much just need to install tensorflow and download/install tensorflow/models (and all their dependencies)
