
from keras._tf_keras.keras.models import load_model
from gen_data import generate_training_data_parallel

from process_data import process_data_to_csv
from functions import create_gui

from training_data import training_model

def main(final_model_path):

    try:
        create_gui(final_model_path)
    except:
        file_path = generate_training_data_parallel()
        transformed_data_path = process_data_to_csv(file_path)
        final_model_path = training_model(transformed_data_path)
        create_gui(final_model_path)

if __name__ == "__main__":
    final_model_path = 'final_model.h5'
    main(final_model_path)