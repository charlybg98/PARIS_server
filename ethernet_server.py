import socket
import struct
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import json
from os.path import expanduser, join
import numpy as np

# Configuraciones
BUFFER_SIZE = 4096
MAX_LENGTH = 35
PRED_THRESHOLD = 0.8
IMAGE_SIZE = (224, 360)
WINDOW_SIZE = 10
NUM_FEATURES = 108

# Rutas para cargar modelos y tokenizer
language_model_path = join(expanduser("~"), "models", "ALBERT_model")
image_model_path = join(expanduser("~"), "models", "ImageClassification_model")
sequential_model_path = join(expanduser("~"), "models", "Sequential_model")
tokenizer_path = join(expanduser("~"), "models", "ALBERT_tokenizer")
answers_path = join(expanduser("~"), "models", "answers.json")

# Cargar modelos y tokenizer
language_model = TFAlbertForSequenceClassification.from_pretrained(language_model_path)
image_model = tf.keras.models.load_model(image_model_path)
sequential_model = tf.keras.models.load_model(sequential_model_path)
tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)

with open(answers_path, "r", encoding="utf-8") as json_file:
    answers_dict = json.load(json_file)


def warmup_inferences():
    # Inferencia de calentamiento para el modelo de lenguaje
    for _ in range(3):
        sample_text = " "
        encodings = tokenizer(
            sample_text,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        _ = language_model.predict(
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }
        )

    # Inferencia de calentamiento para el modelo de clasificación de imágenes
    for _ in range(3):
        _ = image_model.predict(np.zeros((1, *IMAGE_SIZE, 3)))

    # Inferencia de calentamiento para el modelo secuencial
    for _ in range(3):
        _ = sequential_model.predict(np.zeros((1, WINDOW_SIZE, NUM_FEATURES)))


warmup_inferences()


def predict_image_label(image_data):
    image = tf.io.decode_png(image_data, channels=3)
    image_resized = tf.image.resize(image, IMAGE_SIZE)
    image_norm = image_resized / 255.0
    image_input = tf.keras.applications.mobilenet.preprocess_input(image_resized)
    input_image = tf.expand_dims(image_input, 0)
    predictions = image_model.predict(input_image, verbose=0)
    predicted_index = tf.argmax(predictions[0]).numpy()
    return predicted_index


def make_language_inference(text_data):
    text = text_data.decode("utf-8")
    encodings = tokenizer(
        text,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    predictions = language_model.predict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        },
        verbose=0,
    )
    predictions = tf.nn.softmax(predictions.logits, axis=-1)
    max_prob = tf.reduce_max(predictions).numpy()
    if max_prob < PRED_THRESHOLD:
        return "En este momento, no dispongo de la información suficiente para proporcionar una respuesta precisa."
    else:
        predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
        return answers_dict.get(str(predicted_class), "Respuesta no encontrada.")


def predict_section(flat_window_data, label_int):
    window_data = np.array(flat_window_data).reshape(WINDOW_SIZE, NUM_FEATURES)
    window_data[
        -1, 102 : 102 + 6
    ] = label_int  # Incorporar Label_int en la posición correcta
    prediction = sequential_model.predict(np.expand_dims(window_data, axis=0))
    predicted_section = np.argmax(prediction, axis=-1)[0]
    return predicted_section


# Configuración del servidor de sockets
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("192.168.1.7", 10000)
sock.bind(server_address)
sock.listen(1)
print("Servidor iniciado en {} puerto {}".format(*server_address))


while True:
    connection, client_address = sock.accept()
    try:
        # Recibir tipo de mensaje (texto o imagen + datos)
        message_type = connection.recv(1).decode("utf-8")

        # Recibir longitud del mensaje
        message_length_data = connection.recv(4)
        message_length = struct.unpack("!I", message_length_data)[0]
        message_data = b""

        while len(message_data) < message_length:
            data = connection.recv(BUFFER_SIZE)
            if not data:
                break
            message_data += data

        # Procesar el mensaje
        if len(message_data) == message_length:
            print("Mensaje recibido correctamente.")
            if message_type == "T":
                response = make_language_inference(message_data)
                connection.sendall(response.encode("utf-8"))
            elif message_type == "I":
                # Procesamiento de imagen y modelo secuencial
                label_int = predict_image_label(message_data)

                # Recibir y procesar los datos de la ventana
                window_length_data = connection.recv(4)
                window_length = struct.unpack("!I", window_length_data)[0]
                window_data = b""
                while len(window_data) < window_length:
                    data = connection.recv(BUFFER_SIZE)
                    if not data:
                        break
                    window_data += data

                if len(window_data) == window_length:
                    flat_window_data = np.frombuffer(window_data, dtype=np.float32)
                    predicted_section = predict_section(flat_window_data, label_int)
                    # Enviar Label_int y Section como enteros
                    connection.sendall(struct.pack("!II", label_int, predicted_section))
                else:
                    print("Error: no se recibieron todos los datos de la ventana.")
            else:
                print("Tipo de mensaje no reconocido")
        else:
            print("Error: no se recibieron todos los datos del mensaje.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    finally:
        connection.close()
