import pandas as pd
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

def process_data_to_csv(file_path):

    logging.info("Processing started.")

    logging.info("Loading data from Excel.")
    data = pd.read_excel(file_path)
    logging.info("Processing %i lines from excel", len(data))
    logging.info("Selecting necessary columns.")
    data = data[["Spot", "Status", "Date", "Period Start", "Period End"]]

    logging.info("Converting columns to datetime.")
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    data["Period Start"] = pd.to_datetime(data["Period Start"], format="%H:%M:%S")
    data["Period End"] = pd.to_datetime(data["Period End"], format="%H:%M:%S")

    logging.info("Adding derived columns.")
    data["Day of Week"] = data["Date"].dt.dayofweek
    data["Start Time (min)"] = data["Period Start"].dt.hour * 60 + data["Period Start"].dt.minute
    data["End Time (min)"] = data["Period End"].dt.hour * 60 + data["Period End"].dt.minute

    def process_chunk(chunk, chunk_id):
        logging.info(f"Processing chunk {chunk_id} with {len(chunk)} rows.")
        expanded_data = []
        for _, row in chunk.iterrows():
            for time in range(row["Start Time (min)"], row["End Time (min)"], 5):
                expanded_data.append({
                    "Spot": row["Spot"],
                    "Date": row["Date"],
                    "Day of Week": row["Day of Week"],
                    "Time (min)": time,
                    "Status": row["Status"]
                })
        logging.info(f"Chunk {chunk_id} processed successfully.")
        return expanded_data

    num_chunks = 24  # Adjust this number based on dataset size and CPU
    logging.info(f"Splitting data into {num_chunks} chunks.")
    data_chunks = np.array_split(data, num_chunks)

    all_data = []
    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = [
            executor.submit(process_chunk, chunk, i)
            for i, chunk in enumerate(data_chunks)
        ]
        for i, future in enumerate(futures):
            chunk_data = future.result()
            all_data.extend(chunk_data)
            logging.info(f"Chunk {i} combined into final results.")

    logging.info("Converting expanded data to DataFrame.")
    expanded_df = pd.DataFrame(all_data)

    output_file = Path("./data/transformed_data.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving transformed data to {output_file}.")
    expanded_df.to_csv(output_file, index=False)

    logging.info("Processing completed successfully.")
    
    return output_file
