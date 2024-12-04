import logging
import os
from dotenv import load_dotenv
import pandas as pd
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import numpy as np

load_dotenv()

parking_opening_time = os.environ['OPENING_TIME']
parking_closing_time = os.environ['CLOSING_TIME']
qtd_days_to_gen_data = os.environ['QTD_DAYS_TO_GEN_DATA']

logging.basicConfig(
    filename='./utils/app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Lista de dias de alta demanda (independente do ano)
high_demand_days = [
    {"month": 11, "day": 25},  # Black Friday
    {"month": 12, "day": 24},  # Christmas Eve
    {"month": 12, "day": 31},  # New Year's Eve
]

# Função para verificar se é feriado ou fim de semana
def is_high_demand_date(current_date):
    """Verifica se uma data está na lista de dias de alta demanda ou fim de semana."""
    if current_date.weekday() >= 5:  # Sábado ou Domingo
        return True
    for special_day in high_demand_days:
        if current_date.month == special_day["month"] and current_date.day == special_day["day"]:
            return True
    return False

# Funções de distribuição de chegadas e durações
def get_arrival_times(opening_time, closing_time, demand_profile):
    total_minutes = int((closing_time - opening_time).total_seconds() / 60)
    minutes = np.arange(0, total_minutes)
    probabilities = demand_profile / demand_profile.sum()
    num_arrivals = np.random.poisson(lam=probabilities.sum() * 10)  # Ajuste o fator multiplicador conforme necessário
    arrival_minutes = np.random.choice(minutes, size=num_arrivals, p=probabilities)
    arrival_times = [opening_time + timedelta(minutes=int(m)) for m in arrival_minutes]
    return sorted(arrival_times)

def get_parking_duration():
    # Duração média de estacionamento em minutos (ajuste conforme necessário)
    mean_duration = 60
    std_duration = 30
    duration = max(5, np.random.normal(loc=mean_duration, scale=std_duration))
    return timedelta(minutes=duration)

def generate_data_for_day(day_offset, start_date, parking_spots, opening_time, closing_time):
    thread_id = threading.get_ident()
    logging.info("[Thread %s] Starting processing of day with offset %s", thread_id, day_offset)
    
    current_date = start_date + timedelta(days=day_offset)
    formatted_date = current_date.strftime('%d/%m/%Y')
    is_high_demand = is_high_demand_date(current_date)
    
    day_data = []

    # Definir perfil de demanda por horário
    total_minutes = int((closing_time - opening_time).total_seconds() / 60)
    demand_profile = np.zeros(total_minutes)

    # Padrões diferentes para dias de semana e fins de semana
    if is_high_demand:
        # Picos em horários específicos (ajuste conforme necessário)
        peak_times = [8*60, 12*60, 18*60]  # 8h, 12h, 18h
        for peak in peak_times:
            demand_profile += np.exp(-0.5 * ((np.arange(total_minutes) - peak)/60)**2)
    else:
        peak_times = [9*60, 17*60]  # 9h, 17h
        for peak in peak_times:
            demand_profile += np.exp(-0.5 * ((np.arange(total_minutes) - peak)/90)**2)

    for spot in parking_spots:
        # Gerar horários de chegada para a vaga atual
        arrival_times = get_arrival_times(opening_time, closing_time, demand_profile)
        occupied_periods = []

        for arrival_time in arrival_times:
            departure_time = arrival_time + get_parking_duration()
            if departure_time > closing_time:
                departure_time = closing_time
            occupied_periods.append((arrival_time, departure_time))

        # Combinar e ordenar períodos
        occupied_periods.sort(key=lambda x: x[0])

        # Consolidar períodos sobrepostos
        consolidated_periods = []
        for period in occupied_periods:
            if not consolidated_periods:
                consolidated_periods.append(period)
            else:
                last_period = consolidated_periods[-1]
                if period[0] <= last_period[1]:
                    consolidated_periods[-1] = (last_period[0], max(last_period[1], period[1]))
                else:
                    consolidated_periods.append(period)

        # Adicionar períodos ocupados
        for start_time, end_time in consolidated_periods:
            day_data.append({
                "Spot": spot,
                "Status": '0',
                "Date": formatted_date,
                "Period Start": start_time.strftime("%H:%M:%S"),
                "Period End": end_time.strftime("%H:%M:%S"),
            })

        # Calcular períodos livres
        free_periods = []
        previous_end = opening_time

        for start, end in consolidated_periods:
            if previous_end < start:
                free_periods.append((previous_end, start))
            previous_end = end
        if previous_end < closing_time:
            free_periods.append((previous_end, closing_time))

        # Adicionar períodos livres
        for free_start, free_end in free_periods:
            if free_start >= free_end:
                continue

            day_data.append({
                "Spot": spot,
                "Status": '1',
                "Date": formatted_date,
                "Period Start": free_start.strftime("%H:%M:%S"),
                "Period End": free_end.strftime("%H:%M:%S"),
            })

    logging.info(f"[Thread {thread_id}] Finalizado processamento do dia {formatted_date}.")
    return day_data

def generate_training_data_parallel():
    logging.info("Start of script execution with parallelism!")
    
    parking_spots = [f"{letter}{number}" for letter in "ABC" for number in range(1, 17)]
    start_date = datetime.strptime('27/11/2021', '%d/%m/%Y')

    opening_time = datetime.strptime(parking_opening_time, "%H:%M:%S")
    closing_time = datetime.strptime(parking_closing_time, "%H:%M:%S")

    total_days = int(qtd_days_to_gen_data)
    all_data = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                generate_data_for_day,
                day_offset,
                start_date,
                parking_spots,
                opening_time,
                closing_time
            ) for day_offset in range(total_days)
        ]

        for future in futures:
            all_data.extend(future.result())

    output_file = Path('./data/training_data.xlsx')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_data)

    # Removendo colunas desnecessárias
    # df = df.drop(columns=["Period Duration"])

    df.to_excel(output_file, index=False)

    logging.info("Training data successfully generated!")
    logging.info("=" * 50)

if __name__ == "__main__":
    generate_training_data_parallel()



"""
Você é um engenheiro de machine learning renomado, com mais de 25 anos de experiencia na area e já tendo treinado os mais compelxos modelos de ia possível, ao receber um algoritimo de rede neural responsável por prever a fdisponibilidade das vagas de estacionamento em um local com 48 vagas, notou a grande complexidade do algoritimo fornecido, além do mesmo não estar atingindo uma boa assertividade. Tendo isso em vista, você deve refatorar o algoritimo, tornando-o mais simples possível, e melhorando sua assertividade. Sabendo que a atual estrutura dos dados é:

Spot,Date,Day of Week,Time (min),Status
A1,2021-11-27,5,370,0
A1,2021-11-27,5,375,0
A1,2021-11-27,5,380,0
A1,2021-11-27,5,385,0
A1,2021-11-27,5,390,0
A1,2021-11-27,5,395,0
A1,2021-11-27,5,400,0
A1,2021-11-27,5,405,0
A1,2021-11-27,5,410,0
A1,2021-11-27,5,415,0
A1,2021-11-27,5,420,0
A1,2021-11-27,5,425,0
A1,2021-11-27,5,430,0
A1,2021-11-27,5,435,0
A1,2021-11-27,5,440,0

E que a mesma estrutura se repete para todas as 48 vagas, marcando o status da mesma de cinco em cinco minutos para cada data no qual se tem registro (1100 dias para tras do dia 01/12/2024)

E que foi apresentado o seguinte algoritimo para você:

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout, Bidirectional, Layer
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import classification_report
import keras._tf_keras.keras.backend as K
import matplotlib.pyplot as plt
from collections import Counter

# Load and preprocess data
data = pd.read_csv("./data/transformed_data.csv")
optimizer = Adam(learning_rate=0.015)

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

# Visualize and verify data splits
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"  Train indices: {len(train_index)} ({train_index[0]} to {train_index[-1]})")
    print(f"  Test indices: {len(test_index)} ({test_index[0]} to {test_index[-1]})")
    plt.figure(figsize=(10, 1))
    plt.plot(train_index, [fold] * len(train_index), '|', label="Train")
    plt.plot(test_index, [fold] * len(test_index), '|', label="Test")
    plt.legend(loc="upper right")
    plt.title(f"Fold {fold + 1}")
    plt.show()

    # Use only the first fold for training
    if fold == 0:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Verify data shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Check class distributions
print("Class distribution in training data:")
print(Counter(y_train))
print("Class distribution in testing data:")
print(Counter(y_test))

# Compute class weights for imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))

# Define Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Build LSTM model with Bidirectional layers, Dropout, and Attention
model = Sequential()
# Layer 1: Bidirectional LSTM
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.1))
# Layer 2: LSTM
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.1))
# Layer 3: LSTM with Attention
model.add(LSTM(32, return_sequences=True))
model.add(Attention())
# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Configure callbacks
checkpoint_path = "./checkpoints/model_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.2f}.keras"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    save_weights_only=False,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Start with 50 epochs; early stopping will handle termination
    batch_size=8192,  # Smaller batch size for better gradient updates
    class_weight=class_weights,  # Address imbalance
    callbacks=[checkpoint_callback, lr_scheduler, early_stopping]
)

# Save the final model
final_model_path = './final_model.h5'
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Model accuracy on test data: {accuracy:.2%}")

# Generate classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

Adapte-o conforme o necessário escolhendo a melhor função de ativação, quantas camadas forem necessárias e permitindo treinar o modelo de foirma rapida e eficiente (sabendo que vc possui um ryzen 9 com 24 treadhs e 32gb de ram).

Obs:

- Todos os imports do keras pelo tensorflow devem ser feitos dessa forma:

keras._tf_keras.keras.

"""