[![Python](https://img.shields.io/badge/Python-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy\&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)](https://seaborn.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)

# Machine Learning from Scratch

This repository was created to deepen my understanding of machine learning methods by implementing them from scratch in Python Jupyter Notebooks and comparing their results with the models provided by existing libraries.

## Description

The notebooks are organized to demonstrate the step-by-step process of building and evaluating algorithms. They rely on a set of widely used Python libraries:

- **NumPy** – for vectorized and matrix operations, linear algebra, and numerical routines.
- **Pandas** – for structured data manipulation, preprocessing, and tabular analysis.
- **Matplotlib** and **Seaborn** – for data visualization, including exploratory analysis and graphical representation of algorithm results.
- **Scikit-learn** – for access to datasets, utility functions, and baseline models for validation.
- **PyTorch** – for building, training, and evaluating neural networks, offering automatic differentiation and GPU acceleration for deep learning tasks.

Each notebook typically includes:

1. **Math** – key formulas and theoretical background for the algorithm or evaluation metric.
2. **Implementation** – step-by-step Python code that reproduces the method without relying on high-level machine learning functions.
3. **Datasets** – description and links to datasets used in the experiments.
4. **Visualization** – plots illustrating the behavior of the algorithm, decision boundaries, performance metrics, or error analysis.
5. **Comparison** – evaluation of the custom implementation against Scikit-learn (or other libraries) to verify correctness and performance.

## Contents

| Notebook | Description |
|:--------:|-------------|
| [EDA COVID-19](notebooks/01_eda_covid19.ipynb) | A small exploratory data analysis (EDA) of COVID-19 datasets, focusing on general trends and basic insights from the data. |
| [Linear Regression](notebooks/02_linear_regression.ipynb) | Analysis and implementation of linear regression to identify linear relationships that may influence students’ learning outcomes. |
| [K-Nearest Neighbors (KNN)](notebooks/03_knn.ipynb) | Implementation and evaluation of the K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks. |
| [Principal Component Analysis (PCA)](notebooks/04_pca.ipynb) | Application of PCA for dimensionality reduction and visualization, highlighting how major components capture the key variance in the dataset. |
| [Clustering Algorithms](notebooks/05_clustering.ipynb) | Exploration of clustering methods including **k-Means**, **DBSCAN**, and **Agglomerative Clustering** to identify hidden patterns and group structures in the data. |
| [Decision Tree](notebooks/06_decision_tree.ipynb) | Implementation of Decision Tree classifiers for exoplanet prediction, including visualization of decision paths and feature importance analysis. |
| [Naive Bayes](notebooks/07_naive_bayes.ipynb) | Application of the Naive Bayes algorithm for binary classification, demonstrated through a spam detection task using text preprocessing and probabilistic modeling. |
| [Logistic Regression](notebooks/08_logistic_regression.ipynb) | Implementation of Logistic Regression for binary classification, including polynomial feature transformation on the Two Moons dataset to demonstrate nonlinear decision boundaries. |
| [Ensemble Models](notebooks/09_ensemble_models.ipynb) | Detailed exploration of key ensemble learning methods — **Bagging**, **Boosting**, **Voting**, and **Stacking** — including in-depth analysis of Random Forest and AdaBoost algorithms, demonstrating how combining multiple models enhances accuracy, stability, and generalization. |
| [Boosting Algorithms](notebooks/09a_boosting.ipynb) | Comprehensive comparison and implementation of three major boosting methods — **AdaBoost**, **Gradient Boosting**, and **XGBoost**. The notebook demonstrates how each algorithm sequentially improves weak learners, highlights the mathematical intuition behind their update rules, and compares their performance, regularization strategies, and practical applications on real datasets. |
| [Multilayer Perceptron (MLP)](notebooks/10_mlp.ipynb) | Construction and training of a simple Multilayer Perceptron (MLP) neural network for nonlinear classification, demonstrating backpropagation and the effect of hidden layers on learning. |

## Author

Created by [Denys Bondarchuk](https://github.com/thejvdev). Feel free to reach out or contribute to the project!
