import os

from scipy.io import wavfile
import numpy as np

class DataPreparation:
    @staticmethod
    def generate_prbs(length):
        state = np.random.randint(0, 2, size=16)
        feedback = [15, 13, 12, 10]
        sequence = []
        for _ in range(length):
            next_bit = np.logical_xor.reduce(state[feedback])
            sequence.append(state[-1])
            state = np.roll(state, -1)
            state[0] = next_bit
        return np.array(sequence) * 2 - 1

    @staticmethod
    def generate_noisy_data(input_path, output_path, noise_level):
        samplerate, data = wavfile.read(input_path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)  # Ensure 2D shape
        prbs_noise = DataPreparation.generate_prbs(len(data)) * noise_level
        noisy_data = data + prbs_noise.reshape(data.shape)
        os.makedirs(output_path, exist_ok=True)
        wavfile.write(os.path.join(output_path, "noisy_signal.wav"), samplerate, noisy_data.astype(np.int16))
        return noisy_data, data
