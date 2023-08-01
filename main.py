import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

model = tf.saved_model.load('Models/optimized_model')
func = model.signatures['serving_default']

# Warm-up
dummy_input = tf.random.uniform(shape=(1, 224, 224, 3))
dummy_output = func(dummy_input)

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
    img_array = tf.cast(img_array, dtype='float32')
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
    prediction = func(pImg)['predictions']
    predicted_class, probability = np.argmax(prediction), np.max(prediction)
    return {"class": str(predicted_class), "probability": str(probability)}


@app.post("/predict/")
async def predict_class(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pImg = preprocess_image(image_bytes)
    result = predict(pImg)
    return result
