import matplotlib.pyplot as plt
import numpy as np
class Visualization:
    @staticmethod
    def plot_signals(clean_signal, noisy_signal, denoised_signal):
        plt.figure(figsize=(10, 5))
        plt.plot(clean_signal, label='Clean Signal')
        plt.plot(noisy_signal, label='Noisy Signal')
        plt.plot(denoised_signal, label='Denoised Signal')
        plt.legend()
        plt.title('Signal Comparison')
        plt.show()

    @staticmethod
    def calculate_snr(clean_signal, denoised_signal):
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean((clean_signal - denoised_signal)**2)
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"SNR: {snr:.2f} dB")