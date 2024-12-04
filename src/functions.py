import tkinter as tk
from tkinter import ttk, messagebox
from keras._tf_keras.keras.models import load_model
import numpy as np
from datetime import datetime, timedelta

def predict_availability(model, date, time):
    try:
        day_of_week = datetime.strptime(date, "%d/%m/%Y").weekday()
        time_min = int(time.split(':')[0]) * 60 + int(time.split(':')[1])
        is_weekend = 1 if day_of_week >= 5 else 0
        time_sin = np.sin(2 * np.pi * time_min / 1440)
        time_cos = np.cos(2 * np.pi * time_min / 1440)
        
        predictions = []
        for spot in range(48):
            features = np.array([[day_of_week, time_sin, time_cos, is_weekend, spot]])
            prediction = model.predict(features)
            predictions.append(prediction[0][0] > 0.5)
        return predictions
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao prever disponibilidade: {str(e)}")
        return []

def show_availability():
    date = date_entry.get()
    time = time_entry.get()
    
    try:
        datetime.strptime(date, "%d/%m/%Y")
        datetime.strptime(time, "%H:%M")
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira uma data e hora válidas.")
        return
    
    predictions = predict_availability(model, date, time)
    
    if predictions:
        for i, pred in enumerate(predictions):
            color = "green" if pred else "red"
            canvas.itemconfig(spots[i], fill=color)

def create_gui(final_model_path):
    global date_entry, time_entry, canvas, spots, model

    model = load_model(final_model_path)
    root = tk.Tk()
    root.title("Previsão de Disponibilidade de Vagas")

    tk.Label(root, text="Data (dd/mm/yyyy):").grid(row=0, column=0, padx=5, pady=5)
    date_entry = tk.Entry(root)
    date_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(root, text="Hora (HH:MM):").grid(row=1, column=0, padx=5, pady=5)
    time_entry = tk.Entry(root)
    time_entry.grid(row=1, column=1, padx=5, pady=5)

    ttk.Button(root, text="Prever Disponibilidade", command=show_availability).grid(row=2, column=0, columnspan=2, pady=10)

    canvas = tk.Canvas(root, width=600, height=400)
    canvas.grid(row=3, column=0, columnspan=2)

    spots = []
    x, y = 10, 10
    for i in range(48):
        rect = canvas.create_rectangle(x, y, x+50, y+50, fill="gray", outline="black")
        canvas.create_text(x+25, y+25, text=str(i+1), fill="white")
        spots.append(rect)
        x += 60
        if (i + 1) % 8 == 0:
            x = 10
            y += 60

    root.mainloop()

if __name__ == "__main__":
    create_gui()
