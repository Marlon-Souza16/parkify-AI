import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# Carregar os dados processados
data = pd.read_csv("dados_transformados.csv")

# Amostrar os dados para experimentação inicial (opcional: desative para treinar com todos os dados)
data = data.sample(frac=0.10, random_state=42)  # 20% dos dados

# Enriquecimento das features
data["Time Period"] = pd.cut(data["Time (min)"], bins=[0, 360, 720, 1080, 1440],
                             labels=["Early Morning", "Morning", "Afternoon", "Evening"])
data["Is Weekend"] = (data["Day of Week"] >= 5).astype(int)
data["Spot"] = data["Spot"].astype("category").cat.codes
data["Weekday_Hour"] = data["Day of Week"] * data["Time (min)"]

# Features (X) e target (y)
X = data[["Day of Week", "Time (min)", "Is Weekend", "Spot", "Weekday_Hour"]]
y = data["Status"]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo inicial: XGBoost
model = XGBClassifier(
    max_depth=10,
    n_estimators=100,
    learning_rate=0.1,
    n_jobs=-1,  # Paralelização total
    random_state=42
)

# Hiperparâmetros para RandomizedSearchCV
param_dist = {
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# RandomizedSearchCV para ajustar os hiperparâmetros
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# Melhor modelo
best_model = random_search.best_estimator_

# Previsões no conjunto de teste
y_pred = best_model.predict(X_test)

# Avaliação
print("Melhores hiperparâmetros:", random_search.best_params_)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Salvar o modelo treinado
joblib.dump(best_model, 'xgboost_model.pkl')

# Exemplo de carregamento do modelo
# best_model = joblib.load('xgboost_model.pkl')
