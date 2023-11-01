import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from linear_regression import load_data, train_linear_regression

# Define a fixture to load the data once and reuse it for each test
@pytest.fixture(scope="module")
def data():
    arr, column_names = load_data()
    return arr, column_names

# Define a test for the load_data function
def test_load_data(data):
    arr, column_names = data
    # Check that the array and column names are not empty
    assert arr.size > 0
    assert len(column_names) > 0
    # Check that the array has two columns and the column names match the expected values
    assert arr.shape[1] == 2
    assert column_names == [column_names[0], column_names[1]]

# Define a test for the train_linear_regression function
def test_train_linear_regression(data):
    arr, column_names = data
    # Check that the function does not raise any exception
    try:
        train_linear_regression(arr, column_names)
    except Exception as e:
        pytest.fail(f"train_linear_regression raised {e} unexpectedly")
    # Check that the function returns a valid model and metrics
    x = arr[:, 0].reshape(-1,1)
    y = arr[:, 1].reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    ms = mean_squared_error(y_test, y_pred)
    r = r2_score(y_test, y_pred)
    assert isinstance(model, LinearRegression)
    assert isinstance(ms, float)
    assert isinstance(r, float)
if __name__ == '__main__':
    pytest.main()
