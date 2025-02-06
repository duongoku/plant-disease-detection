from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def get_default_device():
    if torch.cuda.is_available():
        # print("cuda available")
        return torch.device("cuda")
    else:
        # print("cuda not available")
        return torch.device("cpu")


def predict_image(image, model, class_names, device):
    """
    Make prediction for a single image
    Args:
        image: PIL Image object or path to image
    """
    transform = transforms.ToTensor()

    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image

    # Convert RGBA to RGB if needed
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Resize to target dimensions
    img = img.resize((256, 256))

    # Convert to tensor
    img_tensor = transform(img)

    # Ensure we have 3 channels
    if img_tensor.shape[0] != 3:
        if img_tensor.shape[0] == 1:
            # If grayscale, repeat channel 3 times
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif img_tensor.shape[0] == 4:
            # If RGBA, take only first 3 channels
            img_tensor = img_tensor[:3]

    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return class_names[predicted.item()], confidence.item()


def main():
    # Load the saved model
    device = get_default_device()
    checkpoint = torch.load(
        "plant_disease_model_color_01.pth",
        map_location=device,
    )
    # "plant_disease_model_color_gray_seg_01.pth"
    # "plant_disease_model_color_01.pth"

    num_classes = len(checkpoint["class_names"])
    model = CNN_NeuralNet(3, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Example usage
    print("-" * 50)
    modifiers = ["color", "grayscale", "segmented"]
    labels = ["Potato___Late_blight"]
    for modifier in modifiers:
        for label in labels:
            test_data_dir = f"D:\\UET\\GRAD_SCHOOL\\AIE\\project\\plantvillage dataset\\{modifier}\\{label}"
            test_image_count = 100
            accurate_predictions = 0
            print(f"Predicting images from {label} class with {modifier} modifier ...")
            for file_name in os.listdir(test_data_dir)[:test_image_count]:
                image_path = os.path.join(test_data_dir, file_name)
                prediction, _ = predict_image(
                    image_path, model, checkpoint["class_names"], device
                )
                accurate_predictions += 1 if prediction == label else 0
                # print(f"Prediction: {prediction}")
            print(f"Accuracy: {accurate_predictions}/{test_image_count}")
            print("-" * 50)


if __name__ == "__main__":
    main()
