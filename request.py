import requests
from time import time

url = "http://localhost:8000/predict/"
file_path = "imagen.jpg"

start = time()
with open(file_path, "rb") as file:
    start_request = time()
    response = requests.post(url, files={"file": file})
    end_request = time()
    response = response.json()
    end = time()
    print(
        f"Clase: {response['class']} con probabilidad de {float(response['probability'])*100:.4f}%"
    )
    print(f"Tiempo de request: {(end_request-start_request):.4f}s")
    print(f"Tiempo total {(end-start):.4f}")
