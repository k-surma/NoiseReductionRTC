import numpy as np
from tensorflow.keras import models
import optuna

from scripts.noise_reduction import NoiseReduction

# Generate placeholder clean data (e.g., 100 samples of 1024-dimensional signals)
clean_data = np.random.rand(100, 1024)

# Add Gaussian noise to simulate noisy data
noise_level = 0.1  # Adjust the noise level as needed
noisy_data = clean_data + np.random.normal(0, noise_level, clean_data.shape)

def objective(trial):
    num_neurons = trial.suggest_int('num_neurons', 64, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    model = NoiseReduction.build_autoencoder((1024,))
    model.compile(optimizer=models.optimizers.Adam(learning_rate), loss='mse')
    loss = model.fit(noisy_data, clean_data, epochs=3, batch_size=32).history['loss'][-1]
    return loss

def run_optimization():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("Best hyperparameters:", study.best_params)