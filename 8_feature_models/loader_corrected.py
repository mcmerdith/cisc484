from numpy import ndarray
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv

# columns 0-1 are unique ids
# columns 2-8 are date components
# columns 9-16 are station data
# columns 38, 39 are radiation
# columns 55, 61, 65 never change
# columns 25-29, 56-57, 59-60, 62-64 are blank
cols = [30, 31, 32, 33, 34, 35, 36, 37]


# Normalize the data
# Normalize the data to a range of 0 to 1 for better performance of neural networks.
scaler = MinMaxScaler(feature_range=(0, 1))


def normalize_3d(data: ndarray) -> ndarray:
    original_shape = data.shape
    x, y, z = original_shape
    # squash the first 2 dims together and normalize the 3rd
    reshaped = data.reshape(x * y, z)
    normalized = scaler.fit_transform(reshaped)
    return normalized.reshape(x, y, z)


def denormalize_3d(data: ndarray) -> ndarray:
    original_shape = data.shape
    x, y, z = original_shape
    # squash the first 2 dims together and normalize the 3rd
    reshaped = data.reshape(x * y, z)
    normalized = scaler.inverse_transform(reshaped)
    return normalized.reshape(x, y, z)


def conv(x):
    try:
        return float(x)
    except ValueError:
        return 0


# Function to create datasets for training and testing.
# This function segments the data into features (X) and target (Y) with a specified number of time steps.

def create_dataset(filename="dataset.csv", *, input_hours, output_hours):
    dataset = np.loadtxt(filename, delimiter=",", quotechar='"', converters=conv,
                         skiprows=1, usecols=cols)
    input_window = input_hours * 60
    output_window = output_hours * 60
    X, Y = [], []
    for i in range(len(dataset) - input_window - output_window - 1):
        X.append(dataset[i:(i + input_window)])
        Y.append(dataset[(i + input_window)
                 :(i + input_window + output_window)])

    X, y = normalize_3d(np.array(X)), normalize_3d(np.array(Y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = create_dataset(
        input_hours=12, output_hours=3)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
