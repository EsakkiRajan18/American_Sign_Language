'''
USAGE:
python preprocess_image.py --num-images 1200
'''

import os
import cv2
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-images', default=1000, type=int,
    help='number of images to preprocess for each category')
args = vars(parser.parse_args())

print(f"Preprocessing {args['num_images']} from each category...")

# 🔧 Path to your real dataset
root_path = "../../American-sign-Language-main/Final Project/Source Code"

# get all folder names (A, B, C...)
dir_paths = [d for d in os.listdir(root_path)
             if os.path.isdir(os.path.join(root_path, d))]
dir_paths.sort()

# preprocess images
for idx, dir_path in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    class_path = os.path.join(root_path, dir_path)
    all_images = os.listdir(class_path)

    os.makedirs(f"../input/preprocessed_image/{dir_path}", exist_ok=True)

    for i in range(min(args['num_images'], len(all_images))):
        rand_id = random.randint(0, len(all_images) - 1)
        img_path = os.path.join(class_path, all_images[rand_id])
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (224, 224))
        cv2.imwrite(f"../input/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", image)

print("DONE")
