import socket
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np

# Carga el modelo MobileNet
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Configura el socket del servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.1.7', 12345))
server_socket.listen()

print("Esperando conexión...")
connection, client_address = server_socket.accept()

try:
    print(f"Conexión desde {client_address}")

    # Recibe la imagen
    image_data = b''
    while True:
        data = connection.recv(1024)
        if not data:
            break
        image_data += data

    # Carga la imagen y la procesa
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))
    x = np.expand_dims(image, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Realiza la inferencia
    preds = model.predict(x)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
    
    # Envía la respuesta
    label = decoded_preds[0][1]
    connection.sendall(label.encode())

finally:
    connection.close()
