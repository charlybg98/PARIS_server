import socket
import struct
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import json
from os.path import expanduser, join

BUFFER_SIZE = 4096
MAX_LENGTH = 35

model_path = join(
    expanduser("~"), "Documents", "NNGUI_server", "models", "ALBERT_model"
)
tokenizer_path = join(
    expanduser("~"), "Documents", "NNGUI_server", "models", "ALBERT_tokenizer"
)
answers_path = join(
    expanduser("~"), "Documents", "NNGUI_server", "answers.json"
)

model = TFAlbertForSequenceClassification.from_pretrained(model_path)
tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)

with open(answers_path, "r", encoding="utf-8") as json_file:
    answers_dict = json.load(json_file)


def warmup_inferences(model, tokenizer, count=5):
    sample_texts = [
        "Hola",
        "¿Cómo estás?",
        "Cuéntame más",
        "Eso es interesante",
        "Sigue así",
    ]
    for text in sample_texts[:count]:
        encodings = tokenizer(
            text,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        _ = model.predict(
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }
        )


warmup_inferences(model, tokenizer)


def make_inference(text_data):
    new_text = text_data.decode("utf-8")
    new_text_encodings = tokenizer(
        new_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )
    predictions = model.predict(
        {
            "input_ids": new_text_encodings["input_ids"],
            "attention_mask": new_text_encodings["attention_mask"],
        },
        verbose=0,
    )
    predictions = tf.nn.softmax(predictions.logits, axis=-1)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
    predicted_label = answers_dict.get(str(predicted_class))
    return predicted_label


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("192.168.1.7", 10000)
sock.bind(server_address)
sock.listen(1)
print("Servidor iniciado en {} puerto {}".format(*server_address))

while True:
    connection, client_address = sock.accept()
    try:
        print("Conexión desde", client_address)
        text_length_data = connection.recv(4)
        if text_length_data:
            text_length = struct.unpack("!I", text_length_data)[0]
            text_data = b""
            while len(text_data) < text_length:
                data = connection.recv(BUFFER_SIZE)
                if not data:
                    break
                text_data += data

            if len(text_data) == text_length:
                print("Texto recibido correctamente.")
                predicted_label = make_inference(text_data)
                connection.sendall(predicted_label.encode("utf-8"))
            else:
                print("Error: no se recibieron todos los datos de texto.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    finally:
        connection.close()
