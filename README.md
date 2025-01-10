# GravitySpy gravitational waves detection

## Paper

[Arxiv](https://arxiv.org/pdf/1611.04596)

## Data

The data was acquired from [kaggle](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves/data)

## GravitySpy devkit

The devkit on [github](https://github.com/Gravity-Spy/GravitySpy), devkit [documentation](https://gravity-spy.github.io/)

## Instructions

### Download dataset

```bash
python download_data.py --dataset_path ./data --kaggle_username {name} --kaggle_key {key}
```

### Torch models

Create environment

```bash
conda env create -f ./envs/torch_env.yml -n gs_torch_env
conda activate gs_torch_env
```

Train CNN model

```bash
python ./models/torch_model.py
```

### Keras models

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
```

Train and evaluate keras CNN model in [jupiter notebook](./models/keras_model.ipynb)
