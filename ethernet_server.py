import socket
import struct
import tensorflow as tf
from PIL import Image
import io
import json
import time

# Carga el mapeo de índices de clase a etiquetas
class_index_path = tf.keras.utils.get_file(
    'imagenet_class_index.json',
    'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
with open(class_index_path) as json_file:
    class_idx = json.load(json_file)
    class_labels = {int(key): value for key, value in class_idx.items()}

# Carga el modelo optimizado con TensorRT
model = tf.saved_model.load('Models/optimized_model')
func = model.signatures['serving_default']

# Función para preprocesar la imagen
def preprocess_image(img):
    img = tf.cast(img, dtype=tf.float32)
    img_array = tf.image.resize(img, size=(224, 224))
    img_array = tf.expand_dims(img_array, axis=0)
    pImg = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return pImg

# Realiza una inferencia de calentamiento con un tensor de ceros
def warmup_inference():
    # Crea un tensor de entrada de ceros
    input_data = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    func(input_data)

warmup_inference()  # Realiza la inferencia de calentamiento

# Función para realizar la inferencia
def make_inference(image_data):
    # Convierte los datos de imagen en un objeto Image
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')  # Asegúrate de que la imagen está en formato RGB
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Preprocesa la imagen
    preprocessed_image = preprocess_image(image_array)
    
    # Realiza la inferencia
    start_time = time.time()
    preds = func(preprocessed_image)['predictions']
    inference_time = time.time() - start_time
    print(f"Inferencia realizada en {inference_time:.2f} segundos")

    # Obtén la clase con la probabilidad más alta
    predicted_class = tf.argmax(preds[0]).numpy()
    confidence = preds[0][predicted_class].numpy()

    # Obtén la etiqueta de la clase
    label = class_labels[predicted_class][1]  # Usando el mapeo cargado previamente
    
    return label, confidence, inference_time

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

        # Recibe la longitud de la imagen
        image_length_data = connection.recv(4)
        if image_length_data:
            image_length = struct.unpack('!I', image_length_data)[0]
            print(f"Longitud de la imagen: {image_length}")

            # Recibe la imagen basada en la longitud
            image_data = b''
            while len(image_data) < image_length:
                data = connection.recv(1024)
                if not data:
                    break
                image_data += data

            if len(image_data) == image_length:
                print("Imagen recibida correctamente.")

                # Realiza la inferencia
                label, confidence, inference_time = make_inference(image_data)
                print(f"Resultado de la inferencia: {label} con un {confidence*100:.2f}% de probabilidad")

                # Envía el resultado de vuelta al cliente
                result_message = f"La predicción es {label} con un {confidence*100:.2f}% de probabilidad. Tiempo de inferencia: {inference_time:.2f} segundos."
                connection.sendall(result_message.encode('utf-8'))

            else:
                print("Error: no se recibieron todos los datos de la imagen.")
        else:
            print("No se recibió la longitud de la imagen.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

    finally:
        # Limpia la conexión
        connection.close()
