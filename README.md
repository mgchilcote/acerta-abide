# Important: the reason for this fork

This is a fork of the below study that was used as part of my project for a deep learning course.

I have updated code to work with TF 2.8 and Python 3.9.7. This does still rely on compatibility libraries to do so. I like this project and hope to work to make it work natively with TF2 and many changes were made that alter the paradigm of how the autoencoders are built. 

There are many other small changes throughout the .py files to make them work better with my project.

I have added a notebook that organizes some of the commands based on the original readme. The original authors' format of using scripts was sufficient and I don't do much more than call the modified scripts in the notebook and organize the results at the end. 

Code was altered in all of the models to allow the results to be saved as txt files that can be read in later to make tables. 





# ACERTA ABIDE [![DOI](https://zenodo.org/badge/38068726.svg)](https://zenodo.org/badge/latestdoi/38068726)
Deep learning using the [ABIDE data](http://fcon_1000.projects.nitrc.org/indi/abide/)

## Environment Setup

Please be aware that this code is meant to be run with Python 3 under Linux (MacOS may work, Windows probably not).
Download the packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running with separate virtual environments

You can also keep a separate set of python libraries to work with the code (if you are not root, or if you have multiple python set ups in the same machine) by running the following commands.
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Just remember to always activate your separate environment before running the code by running
```bash
source env/bin/activate
```

### Tensorflow Quirks

If you have multiple CUDA driver installations in your computer, you may also need to specify environment variables indicating where your preferred driver is installed as follows:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<path/to>/cuda-8.0/targets/x86_64-linux/lib/
```

Tensorflow will also try to use all available GPUs (and memory) in your system, to prevent that, you need to [mask the other GPUs](http://acceleware.com/blog/cudavisibledevices-masking-gpus) using the ```CUDA_VISIBLE_DEVICES``` environment variable. Its parameter is a list of GPU IDs, where GPU ID is the number you see beside the GPU when you list them (e.g. using ```nvidia-smi```), so command:
```bash
export CUDA_VISIBLE_DEVICES=1
```
will only let Tensorflow see the second GPU available in your bus.


## Data preparation

The first step is to download the dataset:

```bash
python download_abide.py
```

This command will download the preprocessed datasets from Amazon S3.

And compile the dataset into CV folds and by experiment.

```bash
python prepare_data.py \
    --whole \
    --male \
    --threshold \
    --folds 10 \
    cc200
```

## Model training

```bash
python nn.py \
    --whole \
    --male \
    --threshold \
    --folds 10 \
    cc200
```

## Model evaluation

```bash
python nn_evaluate.py \
    --whole \
    --male \
    --threshold \
    --folds 10 \
    --mean \
    cc200
```
