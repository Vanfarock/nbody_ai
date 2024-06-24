import tensorflow as tf


class NBodyRNNModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super(NBodyRNNModel, self).__init__(**kwargs)
        self.hidden_units = 64
        self.num_layers = 2

        self.rnn_layers = [self.create_rnn_layer() for _ in range(self.num_layers)]
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(128, activation="relu")
        self.dense4 = tf.keras.layers.Dense(3)

    def create_rnn_layer(self):
        return tf.keras.layers.LSTM(units=self.hidden_units, return_sequences=True)

    def call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = self.rnn_layers[i](x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return x


def train_rnn_model(X, y, epochs):
    optimizer = tf.keras.optimizers.Adam()

    model = NBodyRNNModel()
    model.compile(optimizer=optimizer, loss="mse")
    training = model.fit(X, y, epochs=epochs, validation_split=0.15, shuffle=True)

    model.save("models/rnn.keras")

    return model, training
