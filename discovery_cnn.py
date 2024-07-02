import tensorflow as tf


class NBodyCNNModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NBodyCNNModel, self).__init__(**kwargs)
        self.num_filters = 32
        self.kernel_size = 3
        self.num_layers = 2

        self.conv_layers = [
            tf.keras.layers.Conv1D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                activation="relu",
                padding="same",
            )
            for _ in range(self.num_layers)
        ]
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(64, activation="relu")
        self.dense4 = tf.keras.layers.Dense(6)


def train_cnn_model(X, y, epochs):
    optimizer = tf.keras.optimizers.Adam()

    model = NBodyCNNModel()
    model.compile(optimizer=optimizer, loss="mse")
    training = model.fit(X, y, batch_size=1_000, validation_split=0.15, epochs=epochs)

    model.save("models/discovery_cnn.keras")

    return model, training
