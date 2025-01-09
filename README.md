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
conda env create -f environment.yml -n gs_env
```

Download dataset

```bash
conda activate gs_env
python download_data.py
```

Train model

```bash
conda activate gs_env
python model.py
```
