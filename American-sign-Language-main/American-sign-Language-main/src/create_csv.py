import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths

# 🔧 ABSOLUTE DATASET PATH (your real images)
DATASET_PATH = r"C:\Users\ravi1\Downloads\American_Sign_Language-main\American_Sign_Language-main\American-sign-Language-main\American-sign-Language-main\Final Project\Source Code\AtoZ_3.1"

# get all the image paths
image_paths = list(paths.list_images(DATASET_PATH))
print("TOTAL IMAGES FOUND:", len(image_paths))

if len(image_paths) == 0:
    raise Exception("❌ No images found. Check DATASET_PATH.")

# create a DataFrame
data = pd.DataFrame()

labels = []
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    label = os.path.basename(os.path.dirname(image_path))
    data.loc[i, 'image_path'] = image_path
    labels.append(label)

labels = np.array(labels)

# one hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(labels)}")

for i in range(len(labels)):
    index = np.argmax(labels[i])
    data.loc[i, 'target'] = int(index)

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# save as CSV file
os.makedirs("../input", exist_ok=True)
data.to_csv('../input/data.csv', index=False)

# pickle the binarized labels
os.makedirs("../outputs", exist_ok=True)
print('Saving the binarized labels as pickled file')
joblib.dump(lb, '../outputs/lb.pkl')

print(data.head(10))
