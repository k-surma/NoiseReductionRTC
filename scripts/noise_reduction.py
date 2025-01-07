from tensorflow.keras import layers, models
class NoiseReduction:
    @staticmethod
    def build_autoencoder(input_shape):
        input_signal = layers.Input(shape=input_shape)
        x = layers.Dense(128, activation='relu')(input_signal)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        output_signal = layers.Dense(input_shape[0], activation='linear')(x)
        return models.Model(inputs=input_signal, outputs=output_signal)

    @staticmethod
    def train_model(model, noisy_data, clean_data, epochs=10):
        model.compile(optimizer='adam', loss='mse')
        model.fit(noisy_data, clean_data, epochs=epochs, batch_size=32, validation_split=0.2)
        model.save('models/noise_reduction_model.h5')
        return model