# prepare_dataset.py
# Script for preparing the dataset from Chest X-Ray (Pneumonia) Kaggle
# Creates : lung_data/train | val | test/normal | pneumonia
# 670 images per class (normal and pneumonia)
# 70% train (469), 15% val (101), 15% test (100) per class
#
import os
import shutil
from pathlib import Path
import random


random.seed(42)

# chest_xray/chest_xray/train/NORMAL, train/PNEUMONIA, test/NORMAL, test/PNEUMONIA
raw_dir = Path('chest_xray/chest_xray')
data_dir = Path('lung_data')

# Creating a folder structure
for split in ['train', 'val', 'test']:
    for cls in ['normal', 'pneumonia']:
        os.makedirs(data_dir / split / cls, exist_ok=True)


normal_paths = list((raw_dir / 'train' / 'NORMAL').glob('*.jpeg')) + \
               list((raw_dir / 'test' / 'NORMAL').glob('*.jpeg'))

pneumonia_paths = list((raw_dir / 'train' / 'PNEUMONIA').glob('*.jpeg')) + \
                  list((raw_dir / 'test' / 'PNEUMONIA').glob('*.jpeg'))

# Cutting down to 670 images
random.shuffle(normal_paths)
random.shuffle(pneumonia_paths)

normal_paths = normal_paths[:670]
pneumonia_paths = pneumonia_paths[:670]

# Numbers of images
train_per_class = 469   # ~70%
val_per_class = 101     # ~15%
test_per_class = 100    # ~15%

def copy_images(paths, class_name):
    for i, path in enumerate(paths):
        if i < train_per_class:
            split = 'train'
        elif i < train_per_class + val_per_class:
            split = 'val'
        else:
            split = 'test'
        
        dest_path = data_dir / split / class_name / path.name
        shutil.copy(path, dest_path)


print("Copying normal...")
copy_images(normal_paths, 'normal')

print("Copying pneumonia...")
copy_images(pneumonia_paths, 'pneumonia')

# Summary
print(f"Path: {data_dir.resolve()}")
print(f"Per Class: train {train_per_class}, val {val_per_class}, test {test_per_class}")
print(f"Total: { (train_per_class + val_per_class + test_per_class) * 2 }")
print("\nStructure:")
print("lung_data/")
print("  train/   normal (469), pneumonia (469)")
print("  val/     normal (101), pneumonia (101)")
print("  test/    normal (100), pneumonia (100)")