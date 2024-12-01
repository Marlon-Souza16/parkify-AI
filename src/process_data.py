import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    filename='./utils/app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Início do processamento.")

# Carregar os dados do Excel
logging.info("Carregando dados do Excel.")
data = pd.read_excel("./data/training_data.xlsx")

# Selecionar apenas colunas necessárias
logging.info("Selecionando colunas necessárias.")
data = data[["Spot", "Status", "Date", "Period Start", "Period End"]]

# Converter as colunas para datetime
logging.info("Convertendo colunas para datetime.")
data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
data["Period Start"] = pd.to_datetime(data["Period Start"], format="%H:%M:%S")
data["Period End"] = pd.to_datetime(data["Period End"], format="%H:%M:%S")

# Adicionar colunas derivadas
logging.info("Adicionando colunas derivadas.")
data["Day of Week"] = data["Date"].dt.dayofweek
data["Start Time (min)"] = data["Period Start"].dt.hour * 60 + data["Period Start"].dt.minute
data["End Time (min)"] = data["Period End"].dt.hour * 60 + data["Period End"].dt.minute

# Função para processar um chunk de dados
def process_chunk(chunk, chunk_id):
    logging.info(f"Processando chunk {chunk_id} com {len(chunk)} linhas.")
    expanded_data = []
    for _, row in chunk.iterrows():
        for time in range(row["Start Time (min)"], row["End Time (min)"]):
            expanded_data.append({
                "Spot": row["Spot"],
                "Date": row["Date"],
                "Day of Week": row["Day of Week"],
                "Time (min)": time,
                "Status": row["Status"]
            })
    logging.info(f"Chunk {chunk_id} processado com sucesso.")
    return expanded_data

# Dividir os dados em chunks
num_chunks = 12  # Ajuste este número conforme o tamanho do dataset e a CPU
logging.info(f"Dividindo dados em {num_chunks} chunks.")
data_chunks = np.array_split(data, num_chunks)

# Processar os chunks em paralelo
all_data = []
with ThreadPoolExecutor(max_workers=num_chunks) as executor:
    futures = [
        executor.submit(process_chunk, chunk, i)
        for i, chunk in enumerate(data_chunks)
    ]
    for i, future in enumerate(futures):
        chunk_data = future.result()
        all_data.extend(chunk_data)
        logging.info(f"Chunk {i} combinado aos resultados finais.")

# Converter para DataFrame
logging.info("Convertendo dados expandidos para DataFrame.")
expanded_df = pd.DataFrame(all_data)

# Salvar o dataset transformado em um novo CSV
output_file = Path("./data/dados_transformados.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
logging.info(f"Salvando dados transformados em {output_file}.")
expanded_df.to_csv(output_file, index=False)

logging.info("Processamento concluído com sucesso.")
