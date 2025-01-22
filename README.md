# GravitySpy gravitational waves detection

## Paper

[Arxiv](https://arxiv.org/pdf/1611.04596)

## Data

The data was acquired from [kaggle](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves/data)

## GravitySpy devkit

The devkit on [github](https://github.com/Gravity-Spy/GravitySpy), devkit [documentation](https://gravity-spy.github.io/)

## Instructions

In this project we created different ML models:

- torch lightning model in a .ipynb notebook
- tensorflow keras model in a .ipynb notebook

- torch lightning model in a .py script
- torch model in a .py script

We recommend using the torch lightning notebook

### Download dataset

In orded to download the dataset you need to have an active Kaggle account and create a token in [account settings](https://www.kaggle.com/settings)

Make sure you have free 4GB of storage for the dataset

```bash
python download_data.py --dataset_path ./data --kaggle_username {name} --kaggle_key {token}
```

### Torch and torch lightning models

Create environment

```bash
conda env create -f ./envs/torch_env.yml -n gs_torch_env
conda activate gs_torch_env
```

#### Torch lightning model notebook

Train the torch lightning CNN model in [jupiter notebook](./models/lightning_model.ipynb)

#### Torch lightning model script

Train CNN model using a single script

```bash
python ./models/lightning_model.py
```

#### Torch model script

Train CNN model using a single script

```bash
python ./models/torch_model.py
```

### Tensorflow keras model notebook

Create environment, ignore the numpy version conflicts messages

```bash
conda create -n gs_keras_env python=3.9
conda activate gs_keras_env
pip install tensorflow==2.5.0
pip install matplotlib==3.3.0
pip install pandas
pip install numpy==1.23.0
pip install sklearn
pip install seaborn
pip install ipykernel
```

Train the keras CNN model in [jupiter notebook](./models/keras_model.ipynb)
