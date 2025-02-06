import os
import shutil
import random
from pathlib import Path


def create_train_val_dirs(base_path, disease_classes):
    for split in ["train", "val"]:
        path = os.path.join(base_path, split)
        os.makedirs(path, exist_ok=True)
        for disease in disease_classes:
            os.makedirs(os.path.join(path, disease), exist_ok=True)


def copy_files(files, prefix, src_dir, dst_dir):
    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f"{prefix}_{f}")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def split_dataset(dataset_path, output_path, train_ratio=0.8):
    image_types = ["color", "grayscale", "segmented"]

    # Get all disease classes from color directory (can be any of the three)
    disease_classes = os.listdir(os.path.join(dataset_path, "color"))

    # Create output directories
    create_train_val_dirs(output_path, disease_classes)

    for disease in disease_classes:
        for img_type in image_types:
            images = os.listdir(os.path.join(dataset_path, img_type, disease))

            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            src_dir = os.path.join(dataset_path, img_type, disease)
            train_dir = os.path.join(output_path, "train", disease)
            val_dir = os.path.join(output_path, "val", disease)

            copy_files(train_images, img_type, src_dir, train_dir)
            copy_files(val_images, img_type, src_dir, val_dir)


if __name__ == "__main__":
    dataset_path = "D:\\UET\\GRAD_SCHOOL\\AIE\\project\\plantvillage dataset"  # Change this to your dataset path
    output_path = "split_dataset"  # Change this to your desired output path

    split_dataset(dataset_path, output_path)
    print("Dataset split completed!")
