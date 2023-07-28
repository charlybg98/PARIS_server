from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.utils import img_to_array
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

model = MobileNet(weights="imagenet")

# Warm-up
dummy_input = np.random.rand(1, 224, 224, 3)
dummy_output = model.predict(dummy_input)

app = FastAPI()


def preprocess_image(image_bytes: bytes):
    """
    Preprocesamiento de la imagen.

    Args:
        image_bytes

    Return:
        processed image
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = preprocess_input(img_array)
    return pImg


def predict(pImg: None):
    """
    Procesa la imagen y realiza la predicción:

    Args:
        pImg

    Return:
        Diccionario
    """
    prediction = model.predict(pImg)
    _, predicted_class, probability = decode_predictions(prediction, top=1)[0][0]
    return {"class": str(predicted_class), "probability": str(probability)}


@app.post("/predict/")
async def predict_class(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pImg = preprocess_image(image_bytes)
    result = predict(pImg)
    return result
