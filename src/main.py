import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
data = pd.read_csv("./data/transformed_data.csv")

data["Is Weekend"] = (data["Day of Week"] >= 5).astype(int)
data["Time (sin)"] = np.sin(2 * np.pi * data["Time (min)"] / 1440)
data["Time (cos)"] = np.cos(2 * np.pi * data["Time (min)"] / 1440)

data["Spot"] = data["Spot"].astype("category").cat.codes

X = data[["Day of Week", "Time (sin)", "Time (cos)", "Is Weekend", "Spot"]]
y = data["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Distribuição das classes no conjunto de treinamento:")
print(Counter(y_train))

class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=1024,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

final_model_path = './final_model.h5'
model.save(final_model_path)
print(f"Modelo final salvo em {final_model_path}")

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2%}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
