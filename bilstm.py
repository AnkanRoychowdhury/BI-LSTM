import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from prettytable import PrettyTable
from IPython.display import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential

print("\n")

data_path = "C:\Explainable AI\BI-LSTM\datasetraw"  # Update the path
print('Available data:', os.listdir(data_path))

print("\n")


def read_data(partition):
    data = []
    for fn in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, fn)) as f:
            data.append(pd.read_csv(f, index_col=None))
    return pd.concat(data)


# Divide the main dataset into train (75%), test (15%), val(15%)
# Read the CSV file into a Pandas dataframe
df = pd.read_csv("Human_Data1.csv")  # Update the path

# Calculate the number of rows in the dataframe
num_rows = len(df)

# Calculate the number of rows in the training set (75% of the data)
num_train = int(num_rows * 0.8)

# Calculate the number of rows in the validation set (15% of the data)
num_val = int(num_rows * 0.2)

# Calculate the number of rows in the test set (15% of the data)
num_test = int(num_rows * 0.2)

# Split the data into the training set, validation set, and test set
train = df[:num_train]
val = df[num_train:num_train + num_val]
test = df[num_train + num_val:]

# Save the training set, validation set, and test set to separate CSV files
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

# Reading all data partitions
df_train = read_data('train')
df_test = read_data('test')
df_val = read_data('val')


print(df_train)
print("\n")

# Given data size
print('Train size:', len(df_train))
print('Val size:', len(df_val))
print('Test size:', len(df_test))
print("\n")


def calc_unique_cls(train, test, val):
    """
    Prints the number of unique classes in the data sets.
    """
    train_unq = np.unique(train['family_name'].values)
    val_unq = np.unique(val['family_name'].values)
    test_unq = np.unique(test['family_name'].values)

    print('Number of unique classes in Train:', len(train_unq))
    print('Number of unique classes in Val:', len(val_unq))
    print('Number of unique classes in Test:', len(test_unq))


calc_unique_cls(df_train, df_test, df_val)
print("\n")

# Extract the feature columns
feature_cols = df_train.columns.tolist()
feature_cols.remove('family_name')  # Exclude the target column from features

# Extract the input data
X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values

# Reshape the input data
# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# Verify the preprocessed data
print('Train X shape:', X_train.shape)
print('Val X shape:', X_val.shape)
print('Test X shape:', X_test.shape)

print('Train X : \n', X_train,'\n')
print('Val X : \n', X_val,'\n')
print('Test X : \n', X_test,'\n')


# Integer encoding of family_name
encoder = LabelEncoder()
y = df['family_name'].values
y_encoded = encoder.fit_transform(y)

y_train = encoder.transform(df_train['family_name'].values)
y_test = encoder.transform(df_test['family_name'].values)

print('Train y shape:', y_train.shape)
print('Test y shape:', y_test.shape)


print('y_train:', y_train,'\n')
print('y_test:', y_test,'\n')




# ------------------------------------------ Model: Bidirectional LSTM STARTS-----------------------------------------

n_timesteps = 1
X_train = X_train.reshape((X_train.shape[0], n_timesteps, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], n_timesteps, X_test.shape[1]))

# Build the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=(n_timesteps, X_train.shape[2])))
model.add(Dense(len(encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=1)


# ------------------------------------------ Model: Bidirectional LSTM ENDS-----------------------------------------




print("\nSuccess!")