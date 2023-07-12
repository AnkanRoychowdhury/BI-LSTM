import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Read data from CSV file
data = pd.read_csv('Human_Data2.csv')

# Separate features and labels
X = data.iloc[:, 1:].values  # Input features (all columns except the first)
y = data.iloc[:, 0].values   # Class labels (first column)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]


print('X_train:', X_train.shape,'\n')
print('X_test:', X_test.shape,'\n')
print('y_train:', y_train.shape,'\n')
print('y_test:', y_test.shape,'\n')


# Reshape input data to match the expected input shape
n_timesteps = 1
X_train = X_train.reshape((X_train.shape[0], n_timesteps, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], n_timesteps, X_test.shape[1]))

# Build the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=(n_timesteps, X_train.shape[2])))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Generate loss graph
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Generate confusion matrix
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Calculate classification metrics
report = classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=label_encoder.classes_)
print('Classification Report:')
print(report)