import socket
import struct
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
import json
import time

BUFFER_SIZE = 4096
MAX_LENGTH = 35

# Carga el modelo de lenguaje ALBERT y el tokenizer
model_path = "models/ALBERT_trained"
model = tf.keras.models.load_model(model_path, custom_objects={"TFAlbertModel": TFAlbertModel})

tokenizer_path = 'dccuchile/albert-large-spanish'
tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)

# Carga las respuestas
with open('answers.json', 'r', encoding='utf-8') as json_file:
    answers_dict = json.load(json_file)

# Función para realizar una inferencia de calentamiento
def warmup_inferences(model, count=5):
    warmup_data = {"input_ids": tf.constant([tokenizer.encode("Hola")]), "attention_mask": tf.constant([[1]])}
    for _ in range(count):
        _ = model(warmup_data)

# Realiza inferencias de calentamiento
warmup_inferences(model)

# Función para realizar la inferencia con el modelo de lenguaje
def make_inference(text_data):
    new_text = text_data.decode('utf-8')

    # Preprocesamiento del texto con el tokenizer
    new_text_encodings = tokenizer(new_text,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=MAX_LENGTH,
                                   return_tensors="tf")

    # Realiza la inferencia
    start_time = time.time()
    predictions = model.predict(
        {
            "input_ids": new_text_encodings["input_ids"],
            "attention_mask": new_text_encodings["attention_mask"],
        },
        verbose=0,
    )
    inference_time = time.time() - start_time
    print(f"Inferencia realizada en {inference_time:.2f} segundos")

    # Procesamiento de la predicción
    predictions = tf.nn.softmax(predictions.logits, axis=-1)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
    predicted_label = answers_dict.get(str(predicted_class))

    return predicted_label, inference_time

# Crea un socket TCP/IP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Enlaza el socket a la dirección del servidor y el puerto
server_address = ('192.168.1.7', 10000)
sock.bind(server_address)

# Escucha conexiones entrantes
sock.listen(1)

print('Servidor iniciado en {} puerto {}'.format(*server_address))
print('Esperando para conectarse...')

while True:
    # Espera una conexión
    connection, client_address = sock.accept()

    try:
        print('Conexión desde', client_address)

        # Recibe la longitud del texto
        text_length_data = connection.recv(4)
        if text_length_data:
            text_length = struct.unpack('!I', text_length_data)[0]
            print(f"Longitud del texto: {text_length}")

            # Recibe el texto basado en la longitud
            text_data = b''
            while len(text_data) < text_length:
                data = connection.recv(BUFFER_SIZE)
                if not data:
                    break
                text_data += data

            if len(text_data) == text_length:
                print("Texto recibido correctamente.")

                # Realiza la inferencia
                predicted_label, inference_time = make_inference(text_data)
                print(f"Resultado de la inferencia: {predicted_label}")

                # Envía el resultado de vuelta al cliente
                result_message = f"Respuesta: {predicted_label}. Tiempo de inferencia: {inference_time:.2f} segundos."
                connection.sendall(result_message.encode('utf-8'))

            else:
                print("Error: no se recibieron todos los datos de texto.")
        else:
            print("No se recibió la longitud del texto.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

    finally:
        # Limpia la conexión
        connection.close()
