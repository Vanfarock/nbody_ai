import tensorflow as tf


class NBodyANNModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NBodyANNModel, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(128, activation="relu")
        self.dense4 = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


def train_ann_model(X, y, epochs):
    optimizer = tf.keras.optimizers.Adam()

    model = NBodyANNModel()
    model.compile(optimizer=optimizer, loss="mse")
    training = model.fit(X, y, epochs=epochs, validation_split=0.15, shuffle=True)

    model.save("models/ann.keras")

    return model, training
