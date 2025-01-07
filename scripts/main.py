import numpy as np

from scripts.data_preparation import DataPreparation
from scripts.noise_reduction import NoiseReduction
from scripts.optimization import run_optimization
from scripts.visualization import Visualization

if __name__ == "__main__":
    # Step 1: Prepare Data
    data_path = 'data/raw/test1audioEN.wav'
    processed_path = 'data/processed/'
    noisy_data = DataPreparation.generate_noisy_data(data_path, processed_path, noise_level=0.1)

    # Step 2: Train AI Model
    model = NoiseReduction.build_autoencoder((1024,))
    clean_data = np.random.rand(100, 1024)  # Placeholder for clean data
    NoiseReduction.train_model(model, noisy_data, clean_data)

    # Step 3: Optimize Hyperparameters
    run_optimization()

    # Step 4: Visualize Results
    denoised_signal = model.predict(noisy_data)
    Visualization.plot_signals(clean_data[0], noisy_data[0], denoised_signal[0])
    Visualization.calculate_snr(clean_data[0], denoised_signal[0])
