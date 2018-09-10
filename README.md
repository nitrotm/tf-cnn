# Tensorflow CNN/FCNN architectures

Code to prepare datasets and do model training and evaluation, in the scope of this research project:

[Deep-Learning Image Segmentation - Towards Tea Leaves Harvesting by Automomous Machine](https://sitehepia.hesge.ch/diplome/ITI/2018/ITI_MAT_soir_memoire_diplome_Ducommun_Dit_Boudry_Upegui_2018.pdf)

## Prerequisites

- Python 3.x
- Tensorflow >=1.8 (tested with 1.8 and 1.9)

## How to run

```
python3 src/run.py
```

or by calling predefined tasks in root directory.

## Apps

- bake: create tf-record dataset from picture/mask/weight images
- preview: read and display content of a tf-record dataset
- train: train a single model with given parameters
- trainmulti: train multiple models
- eval: evaluate a model
- predict: report and display model performance on unseen data

## Topologies

Example of model topologies in YAML format are located in `topologies` sub-folder.

## Bake/training configuration

Example of bake and training parameters for single and multiple session are located in `config` sub-folder.

## license

GPLv3
