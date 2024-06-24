import tensorflow as tf


class NBodyCNNModel(tf.keras.Model):
    def __init__(self):
        super(NBodyCNNModel, self).__init__()
        self.num_filters = 32
        self.kernel_size = 3
        self.num_layers = 2

        self.conv_layers = [self.create_conv_layer() for _ in range(self.num_layers)]
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(128, activation="relu")
        self.dense4 = tf.keras.layers.Dense(3)

    def create_conv_layer(self):
        return tf.keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            activation="relu",
            padding="same",
        )

    def call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


def train_cnn_model(X, y, epochs):
    optimizer = tf.keras.optimizers.Adam()

    model = NBodyCNNModel()
    model.compile(optimizer=optimizer, loss="mse")
    training = model.fit(X, y, epochs=epochs, validation_split=0.15, shuffle=True)

    model.save("models/cnn.keras")

    return model, training
