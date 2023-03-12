# PINO applied to 2D wave equations

PINO: Physics Informed Neural Operator

Given a partial differential equations system, PINO allow among to approximate a operator which maps the initial conditions to the final solution of the PDE. 

This project aims at implementating a PINO solver for the 2D wave equation using Modulus.

## Installation

- Clone this repo
```
git clone https://github.com/UgoPelissier/Wave_2D_PINO.git
```

- Install dependencies
```
conda env create -f environment.yml
conda activate modulus-22.09
```

## Creating the dataset

```
cd dataset
python data.py
```

For memory issues, this will first create 10 sets of training/testing pairs, numbered from 0 to 9. Then, it will aggregate them inside the final files.

We'll move the dataset to the working directiry, that we need to create first.

```
(Assuming you're still in dataset/)
cd ..
mkdir outputs && cd outputs && mkdir wave_2d_PINO && mkdir wave_2d_PINO && mkdir dataset
cd ../..
mv dataset/outputs/data/invar_train.pt outputs/wave_2d_PINO/dataset/invar_train.pt
mv dataset/outputs/data/invar_test.pt outputs/wave_2d_PINO/dataset/invar_test.pt
mv dataset/outputs/data/outvar_train.pt outputs/wave_2d_PINO/dataset/outvar_train.pt
mv dataset/outputs/data/outvar_test.pt outputs/wave_2d_PINO/dataset/outvar_test.pt
```

## Running
```
python wave_2d_PINO.py
```
