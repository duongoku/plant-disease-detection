from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from test_predict import CNN_NeuralNet, predict_image, get_default_device
import io
import torch

# Global variables for model and device
device = None
model = None
class_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model, class_names
    device = get_default_device()

    # Load the saved model
    checkpoint = torch.load(
        "plant_disease_model_color_gray_seg_01.pth", map_location=device
    )
    num_classes = len(checkpoint["class_names"])
    model = CNN_NeuralNet(3, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    class_names = checkpoint["class_names"]

    yield

    del model
    del device
    del class_names


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Read and validate the uploaded image
    if not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Make prediction
    try:
        prediction, confidence = predict_image(image, model, class_names, device)

        # Get a more user-friendly disease name by splitting on '___'
        plant, disease = (
            prediction.split("___") if "___" in prediction else (prediction, "")
        )

        response = {
            "prediction": prediction,
            "plant": plant.replace("_", " "),
            "disease": disease.replace("_", " "),
            "confidence": confidence,
        }
        return response
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
