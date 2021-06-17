# Horovod elastic Tensorflow train
## Description
In this example, you will run Horovod with elastic training of a Keras model on the Neu.ro platform.
The goal here is to show how simple it is to run distributed training in an environment where resources (GPU-enabled nodes) might come and go (spot nodes).

The training script is taken from the official [TensorFlow tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner) and is then extended by the corresponding [Horovod tutorial](https://horovod.readthedocs.io/en/stable/elastic.html).

# Running the example
## Preparation
- `pip install -U neuro-cli neuro-flow`
- `neuro login`
- `git clone https://github.com/neuro-inc/mlops-horovod-example.git` - clone this repository
- `cd <local-repo-path>` - switch to the cloned folder
- `ssh-keygen -t rsa -b 4096 -f ssh-keys/id_rsa -q -N ""` - generate a SSH key to allow SSH-based communication between jobs
- `neuro secret add horovod-id-rsa @ssh-keys/id_rsa` - store the private part of the SSH key, which will be used by the main node to coordinate secondary nodes
- `neuro secret add horovod-id-rsa-pub @ssh-keys/id_rsa.pub` - store the public part of the SSH key, which will be used by the secondary nodes to validate the main node

## Launching the main training node
- `neuro-flow run main` - this job will additionally wait for 600 seconds for the secondary nodes to appear (see "--start-timeout 600" in the `main` job's `bash` section).

## Launching secondary nodes
- `neuro-flow run secondary` - execute this command several times to spawn worker nodes. For this example, at least two workers are needed (see "--num-proc 2" in the `main` job's `bash` description).
All of these worker nodes will be connected to the Horovod instance.
You could also update the number of secondary nodes during the training process. Each of them will be connected and synchronized with the main training process - this is the main feature of Horovod in elastic mode.
