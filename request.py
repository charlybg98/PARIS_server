import requests
from time import time
import numpy as np
import ast

# Ruta del archivo
ruta_archivo = "labels.txt"


# Función para leer el archivo y obtener su contenido como cadena de texto
def leer_archivo(ruta):
    with open(ruta, "r") as archivo:
        contenido = archivo.read()
    return contenido


contenido_txt = leer_archivo(ruta_archivo)
labels = ast.literal_eval(contenido_txt)

url = "http://192.168.3.116:8000/predict/"
file_path = "images/test1.jpg"

start = time()
with open(file_path, "rb") as file:
    start_request = time()
    response = requests.post(url, files={"file": file})
    end_request = time()
    response = response.json()
    end = time()
    print(
        f"Clase: {labels[int(response['class'])]} con probabilidad de {float(response['probability'])*100:.4f}%"
    )
    print(f"Tiempo de predicción: {response['time']:.4f}s")
    print(f"Tiempo de request: {(end_request-start_request):.4f}s")
    print(f"Tiempo total {(end-start):.4f}")
