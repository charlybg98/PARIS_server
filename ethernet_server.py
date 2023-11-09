import socket
import struct
import tensorflow as tf
from PIL import Image
import io

# Carga el modelo MobileNetV2
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Función para realizar la inferencia
def make_inference(image_data):
    # Convierte los datos de imagen en un objeto Image
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))  # Tamaño esperado por MobileNetV2

    # Convierte la imagen en un array de numpy y agrega una dimensión de batch
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Añade la dimensión del batch
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)  # Pre-procesamiento específico de MobileNetV2

    # Realiza la inferencia
    predictions = model.predict(image_array)

    # Decodifica las predicciones
    label = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    return label[0][0]

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
                result = make_inference(image_data)
                print("Resultado de la inferencia:", result)

                # Envía el resultado de vuelta al cliente
                connection.sendall(str(result).encode('utf-8'))

            else:
                print("Error: no se recibieron todos los datos de la imagen.")
        else:
            print("No se recibió la longitud de la imagen.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

    finally:
        # Limpia la conexión
        connection.close()
