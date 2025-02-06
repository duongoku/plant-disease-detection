import os
from pathlib import Path


def get_dataset_mapping():
    dataset_dir = Path("D:\\UET\\GRAD_SCHOOL\\AIE\\project\\plantvillage dataset")
    mapping = {}

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if Path(file).suffix.upper() in [".JPG", ".JPEG", ".PNG"]:
                mapping[file] = root.split("\\")[-1]
    return mapping


def rename_test_files():
    test_dir = Path("test_dataset\\test")
    dataset_mapping = get_dataset_mapping()
    label_counters = {}

    for root, _, files in os.walk(test_dir):
        for file in files:
            file_path = Path(root) / file
            if file in dataset_mapping:
                label = dataset_mapping[file]

                if label not in label_counters:
                    label_counters[label] = 1

                new_name = f"{label}_{label_counters[label]}{Path(file).suffix}"
                new_path = Path(root) / new_name

                print(f"Renaming {file_path} to {new_path}")

                file_path.rename(new_path)
                label_counters[label] += 1


if __name__ == "__main__":
    rename_test_files()
