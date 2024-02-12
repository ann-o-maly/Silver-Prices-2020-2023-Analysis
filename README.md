These notebook utilize historical data of silver prices from 2020 to 2023. The data was imported using Yahoo finance. 

[This visualization notebook](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-Visualization.ipynb) contains plots showing the trends of silver prices in the desired time period.

[The linear regression notebook](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-LinearRegression.ipynb) tries to make a prediction using the linear regression model built into Scikit. This is the simplest model covered in this repository.

[Random forest regressor](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-RandomForestRegressor.ipynb) is the next machine learning model that is used in this repository to try to make predictions. 

The deep learning model [LSTM](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-LSTM.ipynb) was then employed for the same purpose with hopes of a better prediction. 

Ultimately, when this fares a worse outcome than the models before, we apply PyCaret to find the best model for this data set and end up with [extra trees regression](https://github.com/ann-o-maly/Silver-Prices-2020-2023-Analysis/blob/main/SilverPrices-Pycaret-ExtraTreesRegression.ipynb) which seems to work the best compared to all the above models that were tested.
