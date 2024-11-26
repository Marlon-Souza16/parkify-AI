import logging
import os
from dotenv import load_dotenv
import pandas as pd
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

import threading

def generate_data_for_day(day_offset, start_date, parking_spots, opening_time, closing_time):
    thread_id = threading.get_ident()
    logging.info("[Thread %s] Starting processing of day with offset %s", thread_id, day_offset)
    
    current_date = start_date + timedelta(days=day_offset)
    formatted_date = current_date.strftime('%d/%m/%Y')
    
    day_data = []

    for spot in parking_spots:
        n = random.randint(2, 7)
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

            day_data.append({
                "Spot": spot,
                "Status": '0',
                "Distance (cm)": distance,
                "Light": light,
                "Date": formatted_date,
                "Period Start": start_time.strftime("%H:%M:%S"),
                "Period End": departure_time.strftime("%H:%M:%S"),
                "Period Duration": str(total_occupation_time)
            })

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

            day_data.append({
                "Spot": spot,
                "Status": '1',
                "Distance (cm)": distance_free,
                "Light": light_free,
                "Date": formatted_date,
                "Period Start": free_start.strftime("%H:%M:%S"),
                "Period End": free_end.strftime("%H:%M:%S"),
                "Period Duration": str(free_end - free_start)
            })

    logging.info(f"[Thread {thread_id}] Finalizado processamento do dia {formatted_date}.")
    return day_data


def generate_training_data_parallel():
    logging.info("Start of script execution with parallelism !")
    
    parking_spots = [f"{letter}{number}" for letter in "ABC" for number in range(1, 17)]
    start_date = datetime.strptime('05/12/2022', '%d/%m/%Y')

    opening_time = datetime.strptime(parking_opening_time, "%H:%M:%S")
    closing_time = datetime.strptime(parking_closing_time, "%H:%M:%S")

    total_days = int(qtd_days_to_gen_data)
    print(total_days)
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
    df.to_excel(output_file, index=False)

    logging.info("Training data sucesffully generated !")
    logging.info("=" * 50)

if __name__ == "__main__":
    generate_training_data_parallel()
