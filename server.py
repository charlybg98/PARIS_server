import socket
import struct
import tensorflow as tf
import numpy as np
import json
from os.path import join, expanduser

BUFFER_SIZE = 4096
IMAGE_SIZE = (224, 360)
trt_model_path = join(expanduser("~"), "Documents", "PARIS", "models", "ImageToAction")
server_config_path = join(expanduser("~"), "Documents", "PARIS", "server_config.json")
warmup_image_path = join(expanduser("~"), "Documents", "PARIS", "warmup.png")

with open(server_config_path, "r") as json_file:
    config = json.load(json_file)
server_ip = config["server_ip"]
server_port = config["server_port"]

image_model = tf.saved_model.load(trt_model_path)


def warmup_inferences():
    input_data = np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32)
    infer = image_model.signatures["serving_default"]
    for _ in range(5):
        _ = infer(tf.constant(input_data))["predictions"]


def predict_image_label(image_data):
    image = tf.io.decode_png(image_data, channels=3)
    image_resized = tf.image.resize(image, IMAGE_SIZE)
    image_input = tf.keras.applications.mobilenet.preprocess_input(image_resized)
    input_image = tf.expand_dims(image_input, 0)

    infer = image_model.signatures["serving_default"]
    predictions = infer(tf.constant(input_image))["predictions"]
    predicted_index = tf.argmax(predictions, axis=-1).numpy()[0]
    return predicted_index


def perform_initial_inference():
    with open(warmup_image_path, "rb") as file:
        image_data = file.read()

    _ = predict_image_label(image_data)


warmup_inferences()
perform_initial_inference()


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (server_ip, server_port)
sock.bind(server_address)
sock.listen(1)
print("Servidor iniciado en {} puerto {}".format(*server_address))

while True:
    connection, client_address = sock.accept()
    try:
        message_length_data = connection.recv(4)
        message_length = struct.unpack("!I", message_length_data)[0]
        message_data = b""

        while len(message_data) < message_length:
            data = connection.recv(BUFFER_SIZE)
            if not data:
                break
            message_data += data

        if len(message_data) == message_length:
            print("Imagen recibida correctamente.")
            label_int = predict_image_label(message_data)
            connection.sendall(struct.pack("!I", label_int))
        else:
            print("Error: no se recibió toda la imagen.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    finally:
        connection.close()
