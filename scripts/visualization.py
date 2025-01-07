import numpy as np


class Visualization:
    @staticmethod
    def calculate_latency(start_time, end_time):
        latency = end_time - start_time
        print(f"Latency: {latency:.2f} seconds")
        return latency

    @staticmethod
    def calculate_packet_loss(original_packets, received_packets):
        loss = (original_packets - received_packets) / original_packets * 100
        print(f"Packet Loss: {loss:.2f}%")
        return loss

    @staticmethod
    def calculate_snr(clean_signal, denoised_signal):
        signal_power = np.mean(clean_signal ** 2)
        noise_power = np.mean((clean_signal - denoised_signal) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"SNR: {snr:.2f} dB")
        return snr
