# Leaf Shapes

Final project for group 48 in the DTU course 02476 MLOps.

*s194354 Line Glade, s153189 Linnea Hjordt Juul & s231733 Dominik Dvoracek*

## Project description

### Goal
The goal of the project is to employ the material presented in this course on a small-scale real-world project, by creating a pipeline for all the stages of a Machine Learning Model's lifecycle including code and data version control, experiment reproducibility, code optimization, testing, debugging and logging. These are all elements of the continuous integration. We also plan to deploy the model using a cloud solution. We aim to monitor the activity of the model after deployment.
We will also examine and evaluate the scalability of the model.

The project in question aims to identify different types of trees by using image classification on images of apple leaves.

### Framework
Along with the Pytorch framework we plan to use the Pytorch Image Models framework (TIMM) as this framework is useful for working with images. Some relevant models are EfficientNet or ResNet 18, but this depends highly on what can be run on the limited resources of our personal computers.

For the structure of the project we used the cookiecutter template, as shown below. We also used dvc for data version control. This list will be updated as we implement more frameworks.

### Data
We are working with the dataset for the [Leaf Classification Competition](https://www.kaggle.com/c/leaf-classification/data) on Kaggle. The dataset consists of a training set with 990 images tree leaves. Additionally, we are given other measures of the leaves, but due to the scope of this project we have chosen to focus only on image classification. There is also a test set consisting of 594 images and features, however we will not be using this set, since this is mainly relevant for participating in the Kaggle Competition. In the training set there are a total of 99 different tree species with 10 images of each species.

### Models
Since we are working with images, an obvious starting point is to create a CNN. For this we can make use of the ResNet18 architecture from TIMM, which is a convolutional neural network architecture and scaling method. However, we are quite restricted by the processing power on our personal computers, as mentioned earlier.


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── leaf_shapes  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
