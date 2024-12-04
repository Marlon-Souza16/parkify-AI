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

high_demand_days = [
    {"month": 11, "day": 25},  # Black Friday
    {"month": 12, "day": 24},  # Christmas Eve
    {"month": 12, "day": 31},  # New Year's Eve
]

def is_high_demand_date(current_date):
    """Verifica se uma data está na lista de dias de alta demanda ou fim de semana."""
    if current_date.weekday() >= 5:
        return True
    for special_day in high_demand_days:
        if current_date.month == special_day["month"] and current_date.day == special_day["day"]:
            return True
    return False

def get_arrival_times(opening_time, closing_time, demand_profile):
    total_minutes = int((closing_time - opening_time).total_seconds() / 60)
    minutes = np.arange(0, total_minutes)
    probabilities = demand_profile / demand_profile.sum()
    num_arrivals = np.random.poisson(lam=probabilities.sum() * 10)
    arrival_minutes = np.random.choice(minutes, size=num_arrivals, p=probabilities)
    arrival_times = [opening_time + timedelta(minutes=int(m)) for m in arrival_minutes]
    return sorted(arrival_times)

def get_parking_duration():
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

    total_minutes = int((closing_time - opening_time).total_seconds() / 60)
    demand_profile = np.zeros(total_minutes)
    if is_high_demand:
        peak_times = [8*60, 12*60, 18*60]  # 8h, 12h, 18h
        for peak in peak_times:
            demand_profile += np.exp(-0.5 * ((np.arange(total_minutes) - peak)/60)**2)
    else:
        peak_times = [9*60, 17*60]
        for peak in peak_times:
            demand_profile += np.exp(-0.5 * ((np.arange(total_minutes) - peak)/90)**2)

    for spot in parking_spots:
        arrival_times = get_arrival_times(opening_time, closing_time, demand_profile)
        occupied_periods = []

        for arrival_time in arrival_times:
            departure_time = arrival_time + get_parking_duration()
            if departure_time > closing_time:
                departure_time = closing_time
            occupied_periods.append((arrival_time, departure_time))

        occupied_periods.sort(key=lambda x: x[0])

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

        for start_time, end_time in consolidated_periods:
            day_data.append({
                "Spot": spot,
                "Status": '0',
                "Date": formatted_date,
                "Period Start": start_time.strftime("%H:%M:%S"),
                "Period End": end_time.strftime("%H:%M:%S"),
            })

        free_periods = []
        previous_end = opening_time

        for start, end in consolidated_periods:
            if previous_end < start:
                free_periods.append((previous_end, start))
            previous_end = end
        if previous_end < closing_time:
            free_periods.append((previous_end, closing_time))

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

    df.to_excel(output_file, index=False)

    logging.info("Training data successfully generated!")
    logging.info("=" * 50)
    
    return output_file
