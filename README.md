Data-driven analysis & prediction of Singapore resale HDB prices

Project Overview

This repository contains a Python Jupyter notebook (Project.ipynb) detailing a machine learning workflow for a regression task. The notebook covers data loading, feature engineering (using custom transformers and scikit-learn pipelines), model training (Linear Regression, Random Forest, and a Multilayer Perceptron), hyperparameter tuning using GridSearchCV, and comprehensive model evaluation, including the generation of key performance metrics and learning curves.

The primary goal of this analysis is to predict a target variable (which is logarithmically transformed in the notebook) and evaluate model performance using Root Mean Squared Error (RMSE) and $R^2$ scores.

Environment Setup (Google Colab)

This project is optimized for use within a Google Colaboratory (Colab) environment. This simplifies setup as core data science packages are pre-installed.

Prerequisites

A Google account (to access Colab).

Setup Steps for Colab

Open the Notebook in Colab:

Upload the Project.ipynb file directly to your Google Drive or Colab session.

Alternatively, open Colab and use File > Upload notebook to load the file.

Install/Update Dependencies: Most key libraries are pre-installed. To ensure compatibility with the notebook's logic, run the following command in the first cell of the notebook:

# Run this cell to install any necessary, non-standard, or updated libraries
!pip install pandas numpy scikit-learn matplotlib seaborn






Dependencies

The project relies on the following major libraries. These are typically available in the standard Colab environment:

Package

Version (Recommended)

Purpose

pandas

>= 2.0.0

Data manipulation and analysis

numpy

>= 1.25.0

Numerical operations and array handling

scikit-learn

>= 1.3.0

Machine learning models, preprocessing, and evaluation

matplotlib

>= 3.7.0

Data visualization (for learning curves)

seaborn

>= 0.12.0

Enhanced statistical data visualization

Data Preparation

The dataset used is related to Housing and can be sourced from the Singapore government's data portal.

Download and Prepare Data File (Manual Upload):

a. Download: Navigate to the official Singapore government data portal:
https://data.gov.sg/datasets?topics=housing&resultId=d_8b84c4ee58e3cfc0ece0d773c8ca6abc
Manually download the dataset file (usually a CSV).

b. Upload to Colab:

Open your Project.ipynb notebook in Colab.

Use the Files icon  on the left sidebar to open the session storage panel.

Click the Upload icon and select the downloaded CSV file. Crucially, rename the file to ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv during or immediately after the upload process. (Note: Files uploaded to Colab session storage are deleted when the runtime disconnects.)

Alternatively, upload the file to your Google Drive and mount the Drive in a Colab cell for persistent storage.

# Load the data after uploading it to the Colab session or Google Drive
import pandas as pd
data = pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')






Reproduction Steps

Follow these steps exactly to reproduce the final figures, tables, and metrics:

Set up Data: Complete the steps in the Data Preparation section above (manually downloading and uploading the ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv file).

Open the Notebook: Ensure the Project.ipynb is open in Google Colab.

Run All Cells: In the Colab interface, navigate to the menu and select: Runtime -> Run all.

Key Results

Upon successful execution, the following key results will be generated and displayed within the notebook output cells:

1. Performance Metrics (Tables)

The final performance of the optimized models (Linear Regression, Random Forest, MLP Regressor) on the test set will be printed in a comparative table format. The reported metrics are:

RMSE (Root Mean Squared Error): Primary metric for evaluating prediction accuracy.

$R^2$ Score: Metric indicating the proportion of the variance in the dependent variable that is predictable from the independent variables.

2. Learning Curve Plots (Figures)

The notebook will display three distinct learning curve figures using the plot_learning_curve function, one for each optimized model:

Figure 1: Learning Curve — Linear Regression

Figure 2: Learning Curve — Random Forest

Figure 3: Learning Curve — MLP Regressor

These plots will help diagnose bias and variance by comparing the Training RMSE and Validation RMSE as the training set size increases.
