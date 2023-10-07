# Loan Status Predictor

## Introduction

The Loan Status Predictor is a machine learning project that uses various algorithms to predict the approval status of loan applications. This README provides an overview of the project, explains its components, and guides you through the steps to use it effectively.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Predicts loan application status (approved or denied) using various machine learning algorithms.
- Handles missing and zero values in the dataset.
- Provides visualization of the confusion matrix for classifier models.
- Includes a selection of machine learning models, including Random Forest, Decision Tree, XGBoost, Gaussian Naive Bayes, and Support Vector Classifier (SVC).

## Getting Started

### Prerequisites

Before using this project, ensure you have the following prerequisites:

- Python (3.x)
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost

You can install the required libraries using `pip`:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost


## Introduction

The Loan Status Predictor is a machine learning project that uses various algorithms to predict the approval status of loan applications. This README provides an overview of the project, explains its components, and guides you through the steps to use it effectively.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Predicts loan application status (approved or denied) using various machine learning algorithms.
- Handles missing and zero values in the dataset.
- Provides visualization of the confusion matrix for classifier models.
- Includes a selection of machine learning models, including Random Forest, Decision Tree, XGBoost, Gaussian Naive Bayes, and Support Vector Classifier (SVC).

## Getting Started

### Prerequisites

Before using this project, ensure you have the following prerequisites:

- Python (3.x)
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost

You can install the required libraries using `pip`:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost```


## Introduction

The Loan Status Predictor is a machine learning project that uses various algorithms to predict the approval status of loan applications. This README provides an overview of the project, explains its components, and guides you through the steps to use it effectively.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Predicts loan application status (approved or denied) using various machine learning algorithms.
- Handles missing and zero values in the dataset.
- Provides visualization of the confusion matrix for classifier models.
- Includes a selection of machine learning models, including Random Forest, Decision Tree, XGBoost, Gaussian Naive Bayes, and Support Vector Classifier (SVC).

## Getting Started

### Prerequisites

Before using this project, ensure you have the following prerequisites:

- Python (3.x)
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost

You can install the required libraries using `pip`:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Installation
1. Clone this repository:
``` git clone https://github.com/rapoorva51/Loan-Status-Prediction.git```
```cd Loan-Status-Prediction```
2. Download the training and test datasets (`train.csv` and `test.csv`) and place them in the project directory.

## Usage
1.  Open a Python environment or a Jupyter Notebook.
2.  Import the necessary libraries:
3. ```  import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```from sklearn.ensemble import RandomForestRegressor, DecisionTreeRegressor,RandomForestClassifier, DecisionTreeClassifier```
```from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix```
```import xgboost as xgb```
```from sklearn.naive_bayes import GaussianNB```
```from sklearn.svm import SVC```
4. Load the training and test datasets:
```` df_train = pd.read_csv("train.csv", encoding='UTF-8')
df_test = pd.read_csv("test.csv", encoding='UTF-8')
 ````
5. Continue with data preprocessing, model training, and model evaluation as described in the provided code.


## Data
The dataset used in this project contains information about loan applications, including applicant income, coapplicant income, loan amount, loan amount term, credit history, and other factors. The dataset is split into training and test sets for model development and evaluation.

## Data preprocessing
Data preprocessing steps include handling missing and zero values, creating dummy variables for categorical columns, and standardizing features for model training.

## Model Training
This project uses various machine learning models, including Random Forest, Decision Tree, XGBoost, Gaussian Naive Bayes, and Support Vector Classifier (SVC). Each model is trained using the training data.

## Model Evaluation
Model evaluation includes calculating accuracy and root mean squared error (RMSE) for regression models and visualizing confusion matrices for classifier models.

## Contributing 
If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request. We welcome contributions and improvements from the community.