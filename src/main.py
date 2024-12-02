import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout

# Load and preprocess data
data = pd.read_csv("./data/transformed_data.csv")

# Feature engineering
data["Is Weekend"] = (data["Day of Week"] >= 5).astype(int)
data["Time (sin)"] = np.sin(2 * np.pi * data["Time (min)"] / 1440)
data["Time (cos)"] = np.cos(2 * np.pi * data["Time (min)"] / 1440)

# Encode categorical features
data["Spot"] = data["Spot"].astype("category").cat.codes

# Define features and target
X = data[["Day of Week", "Time (sin)", "Time (cos)", "Is Weekend", "Spot"]]
y = data["Status"]

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1),  # Reshape for LSTM
    y_train,
    validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
    epochs=10,
    batch_size=32
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(
    X_test.reshape(X_test.shape[0], X_test.shape[1], 1),  # Reshape for LSTM
    y_test,
    verbose=1  # Set verbose=1 if you want to see progress
)

# Print accuracy
print(f"Model accuracy on test data: {accuracy:.2%}")
