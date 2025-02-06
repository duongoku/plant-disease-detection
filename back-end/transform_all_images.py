from PIL import Image
from tqdm import tqdm
import os
import torchvision.transforms as transforms

# Define the transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


def process_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)
        return img_tensor
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def process_and_save_images(input_dir, output_dir):
    """
    Transform and save images while preserving directory structure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get total number of images first
    total_images = sum(
        len([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        for _, _, files in os.walk(input_dir)
    )

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    image_path = os.path.join(root, file)
                    img_tensor = process_image(image_path)

                    if img_tensor is not None:
                        # Get relative path and create destination path
                        rel_path = os.path.relpath(image_path, input_dir)
                        dest_path = os.path.join(output_dir, rel_path)

                        # Create destination directory if needed
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                        # Convert tensor to PIL Image and save immediately
                        img = transforms.ToPILImage()(img_tensor)
                        img.save(dest_path)

                        # Optional: Print shape for verification
                        # print(f"{rel_path}: {img_tensor.shape}")

                    pbar.update(1)


if __name__ == "__main__":
    input_directory = "plantvillage_split"
    output_directory = "plantvillage_split_transformed"
    process_and_save_images(input_directory, output_directory)
