import os
import numpy as np
from scipy.io import wavfile
from tensorflow.keras import models

class DataPreparation:
    @staticmethod
    def generate_noisy_data(input_path, output_path, noise_level):
        samplerate, data = wavfile.read(input_path)
        noise = np.random.normal(0, noise_level, data.shape)
        noisy_data = data + noise
        os.makedirs(output_path, exist_ok=True)
        wavfile.write(os.path.join(output_path, "noisy_signal.wav"), samplerate, noisy_data.astype(np.int16))
        return noisy_data