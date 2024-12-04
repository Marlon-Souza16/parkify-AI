import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, precision_recall_curve, make_scorer, f1_score
from sklearn.utils import class_weight
from collections import Counter
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Carregar e pré-processar os dados
data = pd.read_csv("./data/transformed_data.csv")

# **Engenharia de Features Aprimorada**

# Criar uma feature para a hora do dia
data["Hour"] = data["Time (min)"] // 60

# Criar uma feature para o mês (supondo que a data esteja no formato correto)
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data["Month"] = data["Date"].dt.month

# Adicionar interações entre features
data["Day_Hour"] = data["Day of Week"] * data["Hour"]

# Engenharia de atributos existente
data["Is Weekend"] = (data["Day of Week"] >= 5).astype(int)
data["Time (sin)"] = np.sin(2 * np.pi * data["Time (min)"] / 1440)
data["Time (cos)"] = np.cos(2 * np.pi * data["Time (min)"] / 1440)

# **Codificar variáveis categóricas usando OneHotEncoder**
categorical_features = ["Spot"]
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = onehot_encoder.fit_transform(data[categorical_features])
encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenar as features codificadas com o dataframe original
data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
data = data.drop(columns=categorical_features)

# **Definir features e target**
feature_columns = [
    "Day of Week", "Time (sin)", "Time (cos)", "Is Weekend",
    "Hour", "Month", "Day_Hour"
] + list(encoded_feature_names)

X = data[feature_columns]
y = data["Status"].astype(int)  # Certificar-se de que o target é inteiro

# **Dividir os dados em conjunto de treinamento e teste**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# **Padronizar os dados**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **Reavaliar o balanceamento das classes**

# Verificar distribuição das classes
print("Distribuição das classes no conjunto de treinamento:")
print(Counter(y_train))

# **Não aplicar SMOTE inicialmente e ajustar os pesos das classes**

# Calcular os pesos das classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Pesos das classes:", class_weights)

# **Definir uma função para criar o modelo (necessário para o RandomizedSearchCV)**
def create_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar o modelo
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# **Definir o modelo KerasClassifier para usar com RandomizedSearchCV**
model = KerasClassifier(build_fn=create_model, verbose=0)

# **Definir o espaço de busca para os hiperparâmetros**
param_dist = {
    'optimizer': ['adam', 'rmsprop'],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [512, 1024],
    'epochs': [20, 30],
}

# **Definir o callback de EarlyStopping**
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# **Definir o RandomizedSearchCV**
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring=make_scorer(f1_score),
    cv=3,
    verbose=1,
    random_state=42
)

# **Executar a busca de hiperparâmetros**
random_search.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# **Melhores hiperparâmetros encontrados**
print("Melhores hiperparâmetros:", random_search.best_params_)

# **Treinar o modelo com os melhores hiperparâmetros**
best_model = random_search.best_estimator_.model

# **Avaliar o modelo no conjunto de teste**
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=1)
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2%}")

# **Obter as probabilidades de previsão**
y_scores = best_model.predict(X_test).ravel()

# **Ajustar o threshold com base no F1-Score**
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"Melhor Threshold: {best_threshold}")

# **Aplicar o novo threshold**
y_pred_adjusted = (y_scores >= best_threshold).astype(int)

# **Gerar relatório de classificação**
print(classification_report(y_test, y_pred_adjusted))

# **Salvar o modelo final**
final_model_path = './final_model.h5'
best_model.save(final_model_path)
print(f"Modelo final salvo em {final_model_path}")
