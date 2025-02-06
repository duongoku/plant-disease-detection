from PIL import Image
from anthropic import Anthropic
from test_predict import CNN_NeuralNet, predict_image, get_default_device
import streamlit as st
import torch

client = Anthropic(
    api_key=st.secrets["ANTHROPIC_API_KEY"],
)

cached_answer = {}


def get_answer(plant, disease):
    global cached_answer
    question = f"What is {disease.lower()} in {plant.lower()} and how to treat them? Answer in markdown format with short and consice bullet points."
    if question in cached_answer:
        print("Cached answer")
        return cached_answer[question]

    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model="claude-3-5-sonnet-latest",
    )

    cached_answer[question] = message.content[0].text
    return message.content[0].text


torch.classes.__path__ = []

MODELS = {
    "Standard Model": {
        "path": "back-end/models/plant_disease_model_color_01.pth",
        "description": "General purpose plant disease detection model",
    },
    "Improved Model (can detect grayscale, segmented images)": {
        "path": "back-end/models/plant_disease_model_color_gray_seg_01.pth",
        "description": "Improved model with support for grayscale and segmented images",
    },
}


def load_model(model_name):
    if model_name not in MODELS:
        raise ValueError("Invalid model selection")

    device = get_default_device()
    model_path = MODELS[model_name]["path"]
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(checkpoint["class_names"])
    model = CNN_NeuralNet(3, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model, device, checkpoint["class_names"]


if "model" not in st.session_state or "current_model" not in st.session_state:
    default_model = list(MODELS.keys())[0]
    model, device, class_names = load_model(default_model)
    st.session_state.update(
        {
            "model": model,
            "device": device,
            "class_names": class_names,
            "current_model": default_model,
        }
    )


def analyze_image(image):
    try:
        prediction, confidence = predict_image(
            image,
            st.session_state.model,
            st.session_state.class_names,
            st.session_state.device,
        )

        plant, disease = (
            prediction.split("___") if "___" in prediction else (prediction, "")
        )
        plant = plant.replace("_", " ")
        disease = disease.replace("_", " ")
        confidence_percentage = f"{(confidence * 100):.1f}%"

        st.session_state.last_prediction = prediction
        st.success(
            f"Detected {disease} in {plant} plant "
            f"with {confidence_percentage} confidence."
        )
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")


def main():
    st.set_page_config(page_title="AI Plant Disease Detection", page_icon="🌿")

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f0fff4;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("AI Plant Disease Detection")

    st.markdown(
        "Upload an image of a plant leaf to detect potential diseases using our AI-powered analysis.",
        help="Our AI model will analyze your plant image and identify potential diseases",
    )

    st.markdown("---")

    selected_model = st.selectbox(
        "Select Model",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(st.session_state.current_model),
        help="Choose the AI model for analysis",
    )

    if selected_model != st.session_state.current_model:
        with st.spinner("Loading model..."):
            model, device, class_names = load_model(selected_model)
            st.session_state.update(
                {
                    "model": model,
                    "device": device,
                    "class_names": class_names,
                    "current_model": selected_model,
                }
            )
        st.success(f"Loaded {selected_model}")

    st.info(MODELS[selected_model]["description"])

    st.markdown("---")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded_file = st.file_uploader(
            "Upload an image (Max: 5MB)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG, WebP",
        )

        if uploaded_file is not None:
            if uploaded_file.size > 5 * 1024 * 1024:
                st.error("File size must be under 5MB.")
                return

            image = Image.open(uploaded_file)

            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    analyze_image(image)

            if (
                "last_analyzed_file" not in st.session_state
                or st.session_state.last_analyzed_file != uploaded_file
            ):
                with st.spinner("Analyzing..."):
                    analyze_image(image)
                st.session_state.last_analyzed_file = uploaded_file

    with right_col:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    if "last_prediction" in st.session_state:
        plant, disease = st.session_state.last_prediction.split("___")
        plant = plant.replace("_", " ")
        disease = disease.replace("_", " ")

        if disease.lower() == "healthy":
            st.success(f"Plant is healthy with no disease detected.")
        else:
            st.markdown(
                f"### Plant: {plant.capitalize()} | Disease: {disease.capitalize()}"
            )

            with st.spinner("Getting treatment information..."):
                answer = get_answer(plant, disease)
                st.markdown(answer)


if __name__ == "__main__":
    main()
