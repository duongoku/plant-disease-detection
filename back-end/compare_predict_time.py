from PIL import Image
import os
import time
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
        print("cuda available")
        return torch.device("cuda")
    else:
        print("cuda not available")
        return torch.device("cpu")


def predict_image(image_path, model, class_names, device):
    """
    Make prediction for a single image and measure inference time
    """
    transform = transforms.ToTensor()
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_tensor = transform(img)

    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        inference_time = (time.time() - start_time) * 1000

    return class_names[predicted.item()], inference_time


def compare_inference_times(model, data_dir, class_names, num_images=50):
    """
    Compare inference times between CPU and GPU
    """
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cpu_times = []
    gpu_times = []

    image_files = os.listdir(data_dir)[:num_images]

    # CPU inference
    model = model.to(cpu_device)
    print("\nRunning CPU inference...")
    for file_name in image_files:
        image_path = os.path.join(data_dir, file_name)
        _, inference_time = predict_image(image_path, model, class_names, cpu_device)
        cpu_times.append(inference_time)

    # GPU inference
    if torch.cuda.is_available():
        model = model.to(gpu_device)
        print("\nRunning GPU inference...")
        for file_name in image_files:
            image_path = os.path.join(data_dir, file_name)
            _, inference_time = predict_image(
                image_path, model, class_names, gpu_device
            )
            gpu_times.append(inference_time)

    print("\nPerformance Comparison:")
    print(f"CPU - Average inference time: {sum(cpu_times)/len(cpu_times):.2f} ms")
    if torch.cuda.is_available():
        print(f"GPU - Average inference time: {sum(gpu_times)/len(gpu_times):.2f} ms")
        print(f"Speedup: {sum(cpu_times)/sum(gpu_times):.2f}x")


def main():
    device = get_default_device()
    checkpoint = torch.load("plant_disease_model.pth", map_location=device)

    num_classes = len(checkpoint["class_names"])
    model = CNN_NeuralNet(3, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    data_dir = "D:\\UET\\GRAD_SCHOOL\\AIE\\project\\plantvillage dataset\\grayscale\\Potato___Late_blight"
    compare_inference_times(model, data_dir, checkpoint["class_names"])


if __name__ == "__main__":
    main()
