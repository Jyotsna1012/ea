from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load models
emotion_model = keras.models.load_model("models_raw/FacialExpression_weights.keras")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # 1️⃣ Read uploaded file
    image_bytes = await file.read()

    # 2️⃣ Convert to grayscale 96×96
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((96, 96))
    img_array = np.array(img)

    # 3️⃣ Normalize → (1, 96, 96, 1)
    model_input = img_array.reshape(1, 96, 96, 1).astype("float32") / 255.0

    # 5️⃣ Predict emotion probabilities
    emotion_pred = emotion_model.predict(model_input)[0].tolist()

    # 6️⃣ Optional — get predicted label
    emotion_map = {
        0: 'anger',
        1: 'disgust',
        2: 'sad',
        3: 'happiness',
        4: 'surprise'
    }
    predicted_label = emotion_map[int(np.argmax(emotion_pred))]

    return predicted_label
