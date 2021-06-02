# Horovod elastic Tensorflow train
## Description
In this example you will run Horovod with elastic training of Keras model on the Neu.ro platform.
The goal here is to show how simple is to run a distributed training on platform, where resources (GPU-enabled nodes) might come and go (spot nodes).

The training script is taken from official [TensorFlow tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner) and extended by respective the [Horovod tutorial](https://horovod.readthedocs.io/en/stable/elastic.html).

# Run this example
## Prepare
- `pip install -U neuro-cli neuro-flow`
- `neuro login`
- `git clone` this repository and `cd` into the clonned folder
- `ssh-keygen -t rsa -b 4096 -f ssh-keys/id_rsa -q -N ""` - generate SSH key to allow SSH-based communication between jobs
- `neuro secret add horovod-id-rsa @ssh-keys/id_rsa` - store private part of SSH key, will be used by main node to coordinate secondary nodes
- `neuro secret add horovod-id-rsa-pub @ssh-keys/id_rsa.pub` - store public part of SSH key, which will be used by the secondary nodes to validate main node

## Launch main training node
- `neuro-flow run main` - it will wait for 600 seconds (see "--start-timeout 600" in main job bash section) for secondary nodes to appear.

## Launch secondary nodes
- `neuro-flow run secondary` - execute this command several times to spawn worker nodes. For this example, at least two workers are needed (see "--num-proc 2" in main job bash description).
All of them will be connected to the Horovod instance.
You could also update the number of secondary nodes during the training process. Each of them will be connected and synchronized with the main training process  - thats the main feature of Horovod in elastic mode.
