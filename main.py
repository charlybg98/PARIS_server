import tensorflow as tf
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.utils import img_to_array
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

model = MobileNet(weights="imagenet")
dummy = model.predict(np.random.rand(1, 224, 224, 3), verbose=0)
app = FastAPI()


@app.post("/predict/")
async def predict_class(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = preprocess_input(img_array)

    prediction = model.predict(pImg, verbose=0)
    _, predicted_class, probability = decode_predictions(prediction, top=1)[0][0]
    return {"class": str(predicted_class), "probability": str(probability)}
