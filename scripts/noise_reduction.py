import numpy as np


class NoiseReduction:
    @staticmethod
    def activation_function(x, activation='relu'):
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")

    @staticmethod
    def to_fixed_point(value, scale=1000):
        return np.round(value * scale).astype(int)

    @staticmethod
    def from_fixed_point(value, scale=1000):
        return value / scale

    @staticmethod
    def autoencoder_forward_pass(data, weights, activation='relu', fixed_point=False):
        """Simplified forward pass of an autoencoder implemented manually."""
        if fixed_point:
            data = NoiseReduction.to_fixed_point(data)
            weights['encoder'] = NoiseReduction.to_fixed_point(weights['encoder'])
            weights['decoder'] = NoiseReduction.to_fixed_point(weights['decoder'])

        hidden = NoiseReduction.activation_function(np.dot(data, weights['encoder']), activation)
        output = np.dot(hidden, weights['decoder'])

        if fixed_point:
            output = NoiseReduction.from_fixed_point(output)

        return output

    @staticmethod
    def train_autoencoder(data, noisy_data, epochs=10, learning_rate=0.001, activation='relu', fixed_point=False):
        """Trains an autoencoder with manual backpropagation."""
        input_dim = data.shape[1]
        hidden_dim = 128

        # Initialize weights
        weights = {
            'encoder': np.random.randn(input_dim, hidden_dim) * 0.01,
            'decoder': np.random.randn(hidden_dim, input_dim) * 0.01,
        }

        for epoch in range(epochs):
            # Forward pass
            hidden = NoiseReduction.activation_function(np.dot(noisy_data, weights['encoder']), activation)
            output = np.dot(hidden, weights['decoder'])

            # Compute loss
            loss = np.mean((output - data) ** 2)

            # Backpropagation
            grad_output = 2 * (output - data) / data.shape[0]
            grad_decoder = np.dot(hidden.T, grad_output)
            grad_hidden = np.dot(grad_output, weights['decoder'].T) * (hidden > 0)
            grad_encoder = np.dot(noisy_data.T, grad_hidden)

            # Update weights
            weights['decoder'] -= learning_rate * grad_decoder
            weights['encoder'] -= learning_rate * grad_encoder

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return weights