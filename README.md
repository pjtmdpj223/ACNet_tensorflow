# ACNet_tensorflow
A tensorflow version of ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks
Original author: https://github.com/DingXiaoH/ACNet

Tensorflow 1.15.0
 
When training, replace 'tf.layers.batch_normalization' or other bn function with 'bn_layer' in 'training_bn_and_block.py'
When training, use 'block' and 'block_short' functions in 'training_bn_and_block.py' to create the model.

When testing, 'bn_layer_new' and 'kernel_combine' fuctions in 'testing_bn_and_block.py' should be add to your code.
When testing, replace the 'block' function in 'training_bn_and_block.py' with the 'block' function in 'testing_bn_and_block.py'

Both training and testing, using 'tf.control_dependencies' to update moving mean and moving var. Reference comment part at the bottom.
When defining tf.train.Saver, have to add 'ExponentialMovingAverage' manually. Reference comment part at the bottom.

There is an example of using ACNet Resnet-56 for the cifar-10 date. Run 'git_acnet_training.py' for training. And then, run 'git_acnet_testing.py' for testing.