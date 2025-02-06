# Plant Disease Prediction with CNN (for artificial intelligence systems engineering course - 2425I_INT7024)

This repository is about building an Image classifier CNN with Python for Plant Disease Prediction task.

-   [Plant Disease Prediction with CNN (for artificial intelligence systems engineering course - 2425I_INT7024)](#plant-disease-prediction-with-cnn-for-artificial-intelligence-systems-engineering-course---2425i_int7024)
    -   [CNN Model](#cnn-model)
        -   [Dataset](#dataset)
        -   [Model Architecture](#model-architecture)
        -   [Training](#training)
        -   [Results](#results)
    -   [Prediction Web App](#prediction-web-app)
        -   [Prerequisites](#prerequisites)
        -   [Deployment](#deployment)
    -   [Demo video](#demo-video)

## CNN Model

### Dataset

The dataset used in this project is the PlantVillage dataset which contains images of healthy and diseased plants. The dataset can be downloaded from the following link: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw). Orignal dataset contains 38 classes of plant diseases, with 3 image types: color, grayscale, and segmented images.

### Model Architecture

The original model training code is from [Mohamadreza Momeni on Kaggle](https://www.kaggle.com/code/imtkaggleteam/plant-diseases-detection-pytorch). The model is a Convolutional Neural Network (CNN) with:

-   4 Convolutional blocks (conv1-4) with BatchNorm and ReLU activation
-   2 Residual blocks (res1-2) for better feature extraction
-   MaxPooling layers for dimensionality reduction
-   A classifier with MaxPool, Flatten, and Linear layers
-   Input channels: 3 (RGB images)
-   Output features: 38 (disease classes)

```python
Model Summary:
- conv1: 3 → 64 channels
- conv2: 64 → 128 channels + MaxPool
- res1: 128 → 128 channels (2 conv blocks)
- conv3: 128 → 256 channels + MaxPool
- conv4: 256 → 512 channels + MaxPool
- res2: 512 → 512 channels (2 conv blocks)
- classifier: MaxPool → Flatten → Linear(512, 38)
```

### Training

The model is trained on the PlantVillage dataset with 38 classes of plant diseases. The training code is in the `train.py` file. The model is trained with the following hyperparameters:

-   Batch size: 32
-   Learning rate: 0.01
-   Epochs: 5
-   Weight decay: 0.0001
-   Optimizer: Adam

Training code can be found in [plant-disease-detection.ipynb](back-end/plant-diseases-detection.ipynb) file. There are 2 versions of model checkpoints in the [models](back-end/models) folder. [plant_disease_model_color_01.pth](back-end/models/plant_disease_model_color_01.pth) is trained only on color images, while the other is trained on color, grayscale, and segmented images.

Online training logs can be found in [my Kaggle](https://www.kaggle.com/code/duongoku/plant-diseases-detection).

### Results

At first, the notebook's author model is only trained on color images. The model achieved an accuracy of 99.5% on the test set. The model is then tested on grayscale images and segmented images. The model achieved
an accuracy of 100% on color images (for "Potato Late Blight" disease), but failed to generalize to grayscale images (0% accuracy) and segmented images (7% accuracy). This indicates that the model is highly sensitive to color information and struggles to identify diseases when the color information is removed or modified.

After including the grayscale and segmented images in the training set, the model achieved
much better generalization across all image types. When tested on the "Potato Late Blight" disease class:

-   Color images: 99% accuracy
-   Grayscale images: 97% accuracy
-   Segmented images: 100% accuracy

This demonstrates that training on diverse image representations improves the model's robustness and ability to detect diseases regardless of the image type.

Testing code can be found in [test_predict.ipynb](back-end/test_predict.ipynb) file.

## Prediction Web App

### Prerequisites

-   Python 3.10+
-   Node.js 22+
-   An Anthropic account with Claude AI API key

### Deployment

At first, we try to deploy the front-end using [Next.js](https://nextjs.org/) (in the [front-end](front-end) folder) and the back-end using Flask (in the [back-end](back-end) folder). However, due to the complexity of the online deployment process, we decided to deploy the model using Streamlit. Original front-end code can be found in our another github repository [here](https://github.com/nguyenhungduy/plant-disease-detection).

[Streamlit](https://streamlit.io/) is a Python web framework for building data applications. The deployment code is in the [app.py](back-end/app.py) file. The app includes:

-   Model selection between standard (color-only) and improved (color, grayscale, segmented) versions
-   Image upload support for PNG, JPG, JPEG, WebP formats (max 5MB)
-   Real-time disease detection with confidence scores
-   Disease information and treatment recommendations using Claude AI
-   Caching system for API responses
-   User-friendly interface with clear model descriptions

Features:

-   Dynamic model loading with state management
-   Real-time prediction updates
-   Responsive layout with side-by-side image preview
-   Treatment information in markdown format for easy reading
-   Error handling for invalid uploads
-   Green-themed UI for better visual appeal

The app can be run locally after creating a virtual environment and installing the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Set the Claude AI API key in the `.streamlit/secrets.toml` file (you need to create this file and the folder containing it if it doesn't exist):

```toml
ANTHROPIC_API_KEY = "your_api_key"
```

Then run the app with:

```bash
streamlit run back-end/app.py
```

An online version of the app can be found on [duongoku.streamlit.app](https://duongoku.streamlit.app/).

## Demo video

The demo video can be found in the [demo](demo) folder.
A better quality video can be found on [Youtube](https://youtu.be/19Bfd__M5ic).
[![Plant Disease Prediction with CNN](http://i.ytimg.com/vi/19Bfd__M5ic/hqdefault.jpg)](https://www.youtube.com/watch?v=19Bfd__M5ic)
