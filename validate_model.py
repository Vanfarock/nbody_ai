import tensorflow as tf
from main import evaluate, load_infinity

infinity = load_infinity()
model = tf.keras.models.load_model("models/cnn.keras")
evaluate(model, infinity[0], 100, 0.1, "output/infinito_cnn.txt")
