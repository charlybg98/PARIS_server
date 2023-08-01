import requests
from time import time
import numpy as np
import ast


def leer_archivo(ruta):
    with open(ruta, "r") as archivo:
        contenido = archivo.read()
    return contenido


ruta_archivo = "labels.txt"
url = "http://192.168.3.116:8000/predict/"
image_path = "images/test2.jpg"


contenido_txt = leer_archivo(ruta_archivo)
labels = ast.literal_eval(contenido_txt)

start = time()
with open(image_path, "rb") as file:
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
