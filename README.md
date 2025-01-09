# GravitySpy gravitational waves detection

## Paper

[Arxiv](https://arxiv.org/pdf/1611.04596)

## Data

The data was acquired from [kaggle](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves/data)

## GravitySpy devkit

The devkit on [github](https://github.com/Gravity-Spy/GravitySpy), devkit [documentation](https://gravity-spy.github.io/)

## Commands

Create environment

```bash
conda env create -f env.yml -n gs_env
conda activate gs_env
```

Download dataset

```bash
python download_data.py --dataset_path ./data --kaggle_username {name} --kaggle_key {key}
```

Train model

```bash
python model.py
```
