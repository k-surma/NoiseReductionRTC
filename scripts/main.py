import numpy as np
import time
from aiohttp import web
from scripts.data_preparation import DataPreparation
from scripts.noise_reduction import NoiseReduction
from scripts.visualization import Visualization
from scripts.rtc_streaming import offer

if __name__ == "__main__":
    # Step 1: Setup WebRTC Server
    app = web.Application()
    app.router.add_post("/offer", offer)
    print("WebRTC server running at http://127.0.0.1:8080/offer")
    web.run_app(app, host="127.0.0.1", port=8080)

    # Step 2: Prepare Data
    input_path = r"C:\Users\User\Desktop\projekciki\NoiseReductionRTC\data\raw\test1audioEN.wav"
    output_path = "data/processed/"
    noisy_data, clean_data = DataPreparation.generate_noisy_data(input_path, output_path, noise_level=0.1)

    # Step 3: Train AI Model
    weights = NoiseReduction.train_autoencoder(
        clean_data, noisy_data, epochs=10, learning_rate=0.001, activation="tanh"
    )

    # Step 4: Evaluate Metrics
    start_time = time.time()
    denoised_signal = NoiseReduction.autoencoder_forward_pass(noisy_data, weights, activation="tanh")
    end_time = time.time()

    Visualization.calculate_latency(start_time, end_time)
    Visualization.calculate_packet_loss(original_packets=1000, received_packets=950)
    Visualization.calculate_snr(clean_data[0], denoised_signal[0])
