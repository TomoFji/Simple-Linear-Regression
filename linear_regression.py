import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    df = pd.read_csv('Experience-Salary.csv')
    column_names = df.columns.to_list()
    arr = df.to_numpy()
    return arr, column_names

def train_linear_regression(arr, column_names):
    x = arr[:, 0].reshape(-1,1)
    y = arr[:, 1].reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    ms = mean_squared_error(y_test, y_pred)
    r = r2_score(y_test, y_pred)
    print(f"Mean squared error: {ms:.2f}")
    print(f"R-squared: {r:.2f}")
    plt.scatter(x_test, y_test, color="blue")
    plt.plot(x_test, y_pred, color="red", linewidth=2)
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.title("Linear Regression")
    plt.show()
    
if __name__ == '__main__':
    arr, column_names = load_data()
    train_linear_regression(arr, column_names)