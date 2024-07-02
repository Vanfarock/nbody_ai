import json
import os
import time

import numpy as np
from discovery_ann import train_ann_model
from discovery_cnn import train_cnn_model
from discovery_rnn import train_rnn_model
from matplotlib import pyplot as plt


def load_dataset():
    states = []

    for base_path, _, filenames in os.walk("discovery_data"):
        for filename in sorted(filenames):
            with open(f"{base_path}/{filename}", "r") as file:
                for line in file.readlines():
                    states.append(json.loads(line))

    states = np.array(states)

    print("Total states:", len(states))
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
                    body["vel"][0],
                    body["vel"][1],
                    body["vel"][2],
                    body["mass"],
                ]
            )
        tensor.append(bodies)
    return np.array(tensor)


def convert_tensor_to_train_data(states):
    X = []
    y = []
    for i in range(len(states)):
        if i == 0:
            X.append(states[i])
            continue
        if i == len(states) - 1:
            y.append([body[:-1] for body in states[i]])
            continue
        X.append(states[i])
        y.append([body[:-1] for body in states[i]])
    return np.array(X), np.array(y)


def convert_output_body_to_states(output_body, original_body):
    return {
        "pos": [float(output_body[0]), float(output_body[1]), float(output_body[2])],
        "vel": [float(output_body[3]), float(output_body[4]), float(output_body[5])],
        "mass": float(original_body["mass"]),
    }


def evaluate(model, initial_state, total_steps, filename):
    if os.path.exists(filename):
        os.remove(filename)

    state = initial_state
    for _ in range(total_steps):
        output = model.predict(convert_states_to_tensor([state]))

        with open(filename, "a") as file:
            state = [
                convert_output_body_to_states(body, initial_state[i])
                for i, body in enumerate(output[0])
            ]
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
    states = load_dataset()
    tensor = convert_states_to_tensor(states)
    X, y = convert_tensor_to_train_data(tensor)

    epochs = 200
    evaluation_steps = 300

    start_time = time.time()
    model, training = train_ann_model(X, y, epochs)
    end_time = time.time()
    save_loss(
        training.history["loss"][-1],
        training.history["val_loss"][-1],
        "loss/discovery_ann_loss.txt",
    )
    save_image(
        training.history["loss"],
        training.history["val_loss"],
        "loss/discovery_ann.png",
        "ANN",
    )
    save_training_time(end_time - start_time, "time/discovery_ann.txt")
    evaluate(model, states[0], evaluation_steps, "output/discovery_ann_infinity.json")

    start_time = time.time()
    model, training = train_rnn_model(X, y, epochs)
    end_time = time.time()
    save_loss(
        training.history["loss"][-1],
        training.history["val_loss"][-1],
        "loss/discovery_rnn_loss.txt",
    )
    save_image(
        training.history["loss"],
        training.history["val_loss"],
        "loss/discovery_rnn.png",
        "RNN",
    )
    save_training_time(end_time - start_time, "time/discovery_rnn.txt")
    evaluate(model, states[0], evaluation_steps, "output/discovery_rnn_infinity.json")

    start_time = time.time()
    model, training = train_cnn_model(X, y, epochs)
    end_time = time.time()
    save_loss(
        training.history["loss"][-1],
        training.history["val_loss"][-1],
        "loss/discovery_cnn_loss.txt",
    )
    save_image(
        training.history["loss"],
        training.history["val_loss"],
        "loss/discovery_cnn.png",
        "CNN",
    )
    save_training_time(end_time - start_time, "time/discovery_cnn.txt")
    evaluate(model, states[0], evaluation_steps, "output/discovery_cnn_infinity.json")
