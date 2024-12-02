import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from  keras._tf_keras.keras.models import Sequential
from  keras._tf_keras.keras.layers import Dense, LSTM, Dropout
from  keras._tf_keras.keras.callbacks import ModelCheckpoint

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

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Configure checkpoints
checkpoint_path = "./checkpoints/model_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.2f}.h5"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,       # Path where checkpoints are saved
    save_best_only=True,            # Save only the best model based on validation accuracy
    monitor='val_accuracy',         # Metric to monitor
    mode='max',                     # 'max' for accuracy, 'min' for loss
    save_weights_only=False,        # Save the full model (weights + architecture)
    verbose=1                       # Show logs during saving
)

# Train the model with checkpoints
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=[checkpoint_callback]
)

# Save the final model
final_model_path = './final_model.h5'
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Model accuracy on test data: {accuracy:.2%}")
