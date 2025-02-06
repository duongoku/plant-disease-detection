from pathlib import Path
from typing import List, Tuple
import os
import shutil


if __name__ == "__main__":
    dir = "D:\\UET\\GRAD_SCHOOL\\AIE\\project\\back-end\\plantvillage_split"
    for root, _, files in os.walk(dir):
        for file in files:
            filename_without_ext = Path(file).stem
            if filename_without_ext.endswith(" "):
                # rename file
                old_path = Path(root) / file
                new_path = (
                    Path(root) / filename_without_ext.rstrip() + Path(file).suffix
                )
                print(f"Renaming {old_path} to {new_path}")
                old_path.rename(new_path)
