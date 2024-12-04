import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import precision_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

def training_model(transformed_data_path):

    matplotlib.use('Agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data = pd.read_csv(transformed_data_path)

    data["Is Weekend"] = (data["Day of Week"] >= 5).astype(int)
    data["Time (sin)"] = np.sin(2 * np.pi * data["Time (min)"] / 1440)
    data["Time (cos)"] = np.cos(2 * np.pi * data["Time (min)"] / 1440)

    data["Spot"] = data["Spot"].astype("category").cat.codes

    X = data[["Day of Week", "Time (sin)", "Time (cos)", "Is Weekend", "Spot"]]
    y = data["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

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
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=15,
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

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    precision = precision_score(y_test, y_pred)
    print(f"Precisão do modelo no conjunto de teste: {precision:.2%}")

    print("Matriz de Confusão:")
    confusion_matrix = pd.crosstab(y_test, y_pred.reshape(-1), rownames=['Real'], colnames=['Predito'], margins=True)
    print(confusion_matrix)

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC-ROC: {auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Loss de Treino')
    plt.plot(history.history['val_loss'], label='Loss de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Loss de Treino vs Loss de Validação')
    plt.legend()
    plt.show()
    
    return final_model_path