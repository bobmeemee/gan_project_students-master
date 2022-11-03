# Generative Adversarial Networks

This repository contains the framework of the lab regarding Generative Adversarial Networks. You will create a GAN that
is able to generate anime faces that do not yet exist.

## Prerequisites

### packages

Install the required packages using ```pip install -r requirements.txt```. It is recommended to do this within a virtual
python environment (such as anaconda).

### wandb

Create an account on [Weights and Biases](https://wandb.ai/home). You'll need to create a new project. Call it
GAN_project. Go to the lines containing ``wandb.init(project="GAN_project", entity="YOURNAME")`` and change ``YOURNAME``
to your wandb user name. This line can be found in:

- ``DCGAN.py``

When running one of these files for the first time, wandb will ask you to log in.

### Anime Dataset

1. Create an account on [Kaggle](https://www.kaggle.com/) via your student mail.
2. Download [the anime dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
3. Extract the downloaded files and ensure the folder structure is similar to:
    + Anime_faces
        + class
            + _All images of the dataset_

#### No GPU available

In case no GPU is available, it might be usefull to reduce the size of the dataset. However, this will reduce the
quality of the generated images as well. In a linux terminal you can use the following command:

``find /path/to/dir -type f -print0 | sort -zR | tail -zn +NUMBER_OF_FILES_TO_KEEP | xargs -0 rm``

Here you can choose a number of files to keep by adjusting ``NUMBER_OF_FILES_TO_KEEP``, such as 10,000.

You will notice that training this model takes longer than the AE/VAE of the previous part of the lab. This is due to
the bigger model size and dataset size. Therefore, if no GPU is available, it is recommended to train models overnight.

## Brief file summary

You will find multiple files in the repository, which are explained below.

``custom_parser.py``: a class containing all your options for training/testing your networks. You are allowed to adjust
it by using argparse, but this is not mandatory. You are also allowed to put all your hyperparameters here (perhaps even
using Weights and Biases).

``dataset.py``: this class loads the anime dataset.

``DCGAN.py``: contains the training loop and related code. You will be completing this.

``fid_score.py``: A file that implements the fid-score calculation for PyTorch.

``generate_images.py``: contains functions for generating and saving images.

``models.py``: this class contains all your PyTorch modules (deep networks). It is advised to go through them all. For
more info, see section “Lab”.

``utils.py``: helper functions are defined here; it is advised to go through them. You will also have to complete some
functions!

## Explanation training methodology

Select a name for each model you train using ``--name modelname``. Once training is completed the model is saved in
the ``--save_path path`` directory (experiments/ by default), together with its config file. When evaluating the model
the config file is automatically read and used to load the model. Results during both training and evaluation are
visualised on [Weights and Biases](https://wandb.ai/home).

