# from ann import predict, train_ann_model

# model, training = train_ann_model(epochs=5_000)
# # model = train_ann_model(epochs=1_000)
# # model = tf.keras.models.load_model(
# #     "models/3_body.keras",
# # )
# predict(model, 300)
# print(training.history["loss"])


import json
import os
import random
import time

import numpy as np
from ann import train_ann_model
from cnn import train_cnn_model
from matplotlib import pyplot as plt
from rnn import train_rnn_model


def load_dataset():
    states = []
    forces = []

    for base_path, _, filenames in os.walk("data"):
        for filename in sorted(filenames):
            if filename not in [
                # "lagrange.txt",
                # "forces_lagrange.txt",
                "infinity.txt",
                "forces_infinity.txt",
                "random_0.txt",
                "forces_random_0.txt",
                "random_1.txt",
                "forces_random_1.txt",
                # "random_8.txt",
                # "forces_random_8.txt",
                # "random_9.txt",
                # "forces_random_9.txt",
                # "random_10.txt",
                # "forces_random_10.txt",
            ]:
                continue

            with open(f"{base_path}/{filename}", "r") as file:
                if filename.startswith("forces"):
                    for line in file.readlines():
                        forces.append(json.loads(line))
                else:
                    for line in file.readlines():
                        states.append(json.loads(line))

    random_indices = set()
    while len(random_indices) != 600:
        random_indices.add(random.randrange(0, len(states) - 1))
    random_indices = list(random_indices)

    states = np.array(states)[random_indices]
    forces = np.array(forces)[random_indices]

    print("Total states:", len(states))
    print("Total forces:", len(forces))
    return states, forces


def load_infinity():
    states = []
    for base_path, _, filenames in os.walk("data"):
        for filename in sorted(filenames):
            if filename not in ["infinity.txt"]:
                continue

            with open(f"{base_path}/{filename}", "r") as file:

                for line in file.readlines():
                    states.append(json.loads(line))

    states = np.array(states)
    return states


def convert_states_to_tensor(states):
    tensor = []
    for state in states:
        bodies = []
        for body in state:
            bodies.append(
                [
                    body["pos"][0],
                    body["pos"][1],
                    body["pos"][2],
                    body["mass"],
                ]
            )
        tensor.append(bodies)
    return np.array(tensor)


def evaluate(model, initial_state, total_steps, time_step, filename):
    if os.path.exists(filename):
        os.remove(filename)

    state = initial_state
    for _ in range(total_steps):
        output = model.predict(convert_states_to_tensor([state]))

        new_state = []
        for i, force in enumerate(output[0]):
            body = state[i]
            acceleration = force / body["mass"]
            velocity = [
                body["vel"][0] + acceleration[0] * time_step,
                body["vel"][1] + acceleration[1] * time_step,
                body["vel"][2] + acceleration[2] * time_step,
            ]
            position = [
                body["pos"][0] + velocity[0] * time_step,
                body["pos"][1] + velocity[1] * time_step,
                body["pos"][2] + velocity[2] * time_step,
            ]
            new_state.append(
                {
                    "pos": position,
                    "vel": velocity,
                    "mass": body["mass"],
                }
            )
        state = new_state

        with open(filename, "a") as file:
            file.write(json.dumps(state) + "\n")


def save_loss(training_loss, validation_loss, filename):
    with open(filename, "w") as file:
        file.write("Training loss: " + str(training_loss) + "\n")
        file.write("Validation loss: " + str(validation_loss))


def save_image(training_losses, validation_losses, filename, title):
    X = [i for i in range(len(training_losses))]
    plt.figure(figsize=(10, 5))
    plt.plot(
        X,
        training_losses,
        linestyle="-",
        color="b",
        label="Treinamento",
    )
    plt.plot(
        X,
        validation_losses,
        linestyle="-",
        color="g",
        label="Validação",
    )
    plt.title(title)
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def save_training_time(seconds, filename):
    with open(filename, "w") as file:
        file.write(str(seconds))


if __name__ == "__main__":
    infinity = load_infinity()
    states, forces = load_dataset()
    tensor = convert_states_to_tensor(states)
    X = tensor
    y = forces

    epochs = 2_000
    evaluation_steps = 300

    start_time = time.time()
    model, training = train_ann_model(X, y, epochs)
    end_time = time.time()
    save_loss(
        training.history["loss"][-1],
        training.history["val_loss"][-1],
        "loss/ann_loss.txt",
    )
    save_image(
        training.history["loss"],
        training.history["val_loss"],
        "loss/ann.png",
        "ANN",
    )
    save_training_time(end_time - start_time, "time/ann.txt")
    evaluate(model, infinity[0], evaluation_steps, 0.1, "output/ann_infinity.json")

    start_time = time.time()
    model, training = train_rnn_model(X, y, epochs)
    end_time = time.time()
    save_loss(
        training.history["loss"][-1],
        training.history["val_loss"][-1],
        "loss/rnn_loss.txt",
    )
    save_image(
        training.history["loss"],
        training.history["val_loss"],
        "loss/rnn.png",
        "RNN",
    )
    save_training_time(end_time - start_time, "time/rnn.txt")
    evaluate(model, infinity[0], evaluation_steps, 0.1, "output/rnn_infinity.json")

    start_time = time.time()
    model, training = train_cnn_model(X, y, epochs)
    end_time = time.time()
    save_loss(
        training.history["loss"][-1],
        training.history["val_loss"][-1],
        "loss/cnn_loss.txt",
    )
    save_image(
        training.history["loss"],
        training.history["val_loss"],
        "loss/cnn.png",
        "CNN",
    )
    save_training_time(end_time - start_time, "time/cnn.txt")
    evaluate(model, infinity[0], evaluation_steps, 0.1, "output/cnn_infinity.json")
