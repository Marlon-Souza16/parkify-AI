import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Carregar os dados processados
data = pd.read_csv("dados_transformados.csv")

# Features (X) e target (y)
X = data[["Day of Week", "Time (min)"]]  # Escolha as colunas relevantes para predição
y = data["Status"]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar a árvore de decisão
model = DecisionTreeClassifier(max_depth=10, random_state=42)  # max_depth ajusta a complexidade
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Visualização da árvore de decisão
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=["Day of Week", "Time (min)"], class_names=["Occupied", "Free"], filled=True)
plt.show()
