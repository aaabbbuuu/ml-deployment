from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import keras # Using Keras 3 directly
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MNIST Digit Recognizer API", version="1.0.0")

# Determine model path relative to this file's location
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_NAME = "mnist_model.keras"
MODEL_PATH = MODEL_DIR / MODEL_NAME

model = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}. "
                     "Please ensure the model is trained and placed in the correct directory.")
        # raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        return

    try:
        model = keras.saving.load_model(MODEL_PATH)
        logger.info(f"Model '{MODEL_NAME}' loaded successfully.")
        # model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        # raise RuntimeError(f"Error loading model: {e}")

class ImageData(BaseModel):
    # Expects a 28x28 image flattened into a list of 784 floats (0-1 range)
    image_data: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "image_data": [0.0] * 784 
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(image_input: ImageData):
    if model is None:
        logger.warning("Prediction attempt while model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    if len(image_input.image_data) != 784:
        logger.error(f"Invalid input data length: {len(image_input.image_data)}. Expected 784.")
        raise HTTPException(status_code=400, detail="Input data must be a flattened 28x28 image (784 values).")

    try:
        # The input_shape for the model is (28, 28)
        img_array = np.array(image_input.image_data, dtype=np.float32).reshape(1, 28, 28)

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        probabilities = [float(p) for p in predictions[0]] # Convert numpy floats to Python floats

        logger.info(f"Prediction successful: class={predicted_class}")
        return PredictionResponse(prediction=predicted_class, probabilities=probabilities)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST Digit Recognizer API. Use the /predict endpoint for predictions or see /docs for API documentation."}

# To run this app directly for development (though Dockerfile uses uvicorn command):
# uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
