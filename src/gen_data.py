import logging
import pandas as pd
import random
from datetime import datetime, timedelta

logging.basicConfig(
    filename='./utils/app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Início da execução do script.")
parking_spots = [f"{letter}{number}" for letter in "ABC" for number in range(1, 17)]
start_date = '16/08/2024'

data_columns = {
    "Vaga": [],
    "Status": [],
    "Distância (cm)": [],
    "Luminosidade": [],
    "Data": [],
    "Início do Período": [],
    "Fim do Período": [],
    "Duração do Período": []
}

df = pd.DataFrame(data_columns)
df.to_excel('./data/training_data.xlsx')

opening_time = datetime.strptime("06:00:00", "%H:%M:%S")
closing_time = datetime.strptime("22:00:00", "%H:%M:%S")
start_date_obj = datetime.strptime(start_date, '%d/%m/%Y')

for day_offset in range(100):
    current_date = start_date_obj + timedelta(days=day_offset)
    formatted_date = current_date.strftime('%d/%m/%Y')
    logging.info(f"Processando dados para o dia {formatted_date}.")

    for spot in parking_spots:
        logging.info(f"Iniciando processamento da vaga {spot} para o dia {formatted_date}.")
        n = random.randint(1, 5)
        last_departure_time = opening_time
        occupied_periods = []

        for _ in range(n):
            distance = random.randint(0, 40)
            light = random.randint(600, 1000)
            min_start_time = last_departure_time + timedelta(minutes=random.randint(1, 30))

            if min_start_time >= closing_time:
                break
            start_time = min_start_time

            hour_interval = random.randint(0, 2)
            min_interval = random.randint(0, 59)
            sec_interval = random.randint(0, 59)

            departure_time = start_time + timedelta(
                hours=hour_interval,
                minutes=min_interval,
                seconds=sec_interval
            )

            if departure_time > closing_time:
                break
            total_occupation_time = departure_time - start_time

            occupied_periods.append((start_time, departure_time))

            row_data = {
                "Vaga": spot,
                "Status": '0',
                "Distância (cm)": distance,
                "Luminosidade": light,
                "Data": formatted_date,
                "Início do Período": start_time.strftime("%H:%M:%S"),
                "Fim do Período": departure_time.strftime("%H:%M:%S"),
                "Duração do Período": str(total_occupation_time)
            }

            df.loc[len(df)] = row_data
            logging.info(f"Período ocupado gerado: {row_data}")
            df.to_excel('./data/training_data.xlsx', index=False)
            last_departure_time = departure_time

        free_periods = []
        previous_end = opening_time

        for start, end in occupied_periods:
            if previous_end < start:
                free_periods.append((previous_end, start))
            previous_end = end
        if previous_end < closing_time:
            free_periods.append((previous_end, closing_time))

        for free_start, free_end in free_periods:
            if free_start >= free_end:
                continue

            distance_free = random.randint(40, 400)
            light_free = random.randint(0, 600)

            row_free = {
                "Vaga": spot,
                "Status": '1',
                "Distância (cm)": distance_free,
                "Luminosidade": light_free,
                "Data": formatted_date,
                "Início do Período": free_start.strftime("%H:%M:%S"),
                "Fim do Período": free_end.strftime("%H:%M:%S"),
                "Duração do Período": str(free_end - free_start)
            }

            df.loc[len(df)] = row_free
            logging.info(f"Período livre gerado: {row_free}")
            df.to_excel('./data/training_data.xlsx', index=False)

logging.info("Dados de Treinamento gerados com sucesso.")
