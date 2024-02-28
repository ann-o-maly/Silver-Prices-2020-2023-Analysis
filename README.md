# Silver Price Analysis

This repository contains notebooks analyzing historical silver prices from 2020 to 2023 using various machine learning models.
Note: All notebooks have been created using Google Colab and hence did not require additional downloads.

## Notebooks Overview

- [**Visualization Notebook**](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-Visualization.ipynb): Contains plots showing the trends of silver prices from 2020 to 2023.
- [**Linear Regression Notebook**](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-LinearRegression.ipynb): Attempts to make predictions using the linear regression model built into Scikit-learn.
- [**Random Forest Regressor Notebook**](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-RandomForestRegressor.ipynb): Utilizes the random forest regressor model to predict silver prices.
- [**LSTM Notebook**](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-LSTM.ipynb): Implements a deep learning model (LSTM) for predicting silver prices.
- [**PyCaret Notebook**](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-Pycaret-ExtraTreesRegression.ipynb): Applies PyCaret to find the best model for the dataset, ending up with extra trees regression.

## Usage

Each notebook provides documentation and explanations of the analysis performed and the models used. You can view the notebooks directly on GitHub or download them to run them locally.

## Data Source

The historical silver price data was imported using Yahoo Finance.

## Dependencies

- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Scikit-learn
- PyCaret
- TensorFlow (for LSTM model)

## Outcomes

Various machine learning models and one deep learning model was applied to the data. Each trial resulted in different amounts of accuracy for prediction. 
Ultimately, on using PyCaret, we found that the most suitable method to apply to this particular dataset was Extra Trees Regression; resulting in the highest accuracy for predicting.
