import random

import numpy as np

from scripts.noise_reduction import NoiseReduction

class MonteCarlo:
    @staticmethod
    def optimize(data, noisy_data, trials=10):
        best_loss = float('inf')
        best_params = None

        for _ in range(trials):
            hidden_dim = random.randint(64, 256)
            learning_rate = 10 ** random.uniform(-5, -2)

            weights = NoiseReduction.train_autoencoder(
                data, noisy_data, epochs=5, learning_rate=learning_rate, activation='relu'
            )
            loss = np.mean((NoiseReduction.autoencoder_forward_pass(noisy_data, weights) - data) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_params = {'hidden_dim': hidden_dim, 'learning_rate': learning_rate}

        print(f"Best Parameters: {best_params}, Loss: {best_loss}")
