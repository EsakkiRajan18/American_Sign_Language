'''
USAGE:
python train.py --epochs 10
'''

import pandas as pd
import joblib
import numpy as np
import torch
import random
import albumentations
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import cv2
import cnn_models
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
    help='number of epochs to train the model for')
args = vars(parser.parse_args())

# ---------- SEED ----------
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

seed_everything()
# --------------------------

# set computation device
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}")

# read the data.csv file
df = pd.read_csv('../input/data.csv')
X = df.image_path.values
y = df.target.values

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print(f"Training on {len(xtrain)} images")
print(f"Validating on {len(xtest)} images")

# dataset
class ASLImageDataset(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels
        self.aug = albumentations.Compose([
            albumentations.Resize(224, 224),
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = cv2.imread(self.X[i])
        image = self.aug(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = int(self.y[i])

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


train_data = ASLImageDataset(xtrain, ytrain)
test_data = ASLImageDataset(xtest, ytest)

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)

# model
model = cnn_models.CustomCNN().to(device)
print(model)

# optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# validation
def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == target).sum().item()

    val_loss = running_loss / len(dataloader.dataset)
    val_accuracy = 100. * running_correct / len(dataloader.dataset)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
    return val_loss, val_accuracy

# training
def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    running_correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(dataloader.dataset)
    train_accuracy = 100. * running_correct / len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")
    return train_loss, train_accuracy

# run training
train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
start = time.time()

for epoch in range(args['epochs']):
    print(f"\nEpoch {epoch+1} of {args['epochs']}")
    tl, ta = fit(model, trainloader)
    vl, va = validate(model, testloader)
    train_loss.append(tl)
    train_accuracy.append(ta)
    val_loss.append(vl)
    val_accuracy.append(va)

end = time.time()
print(f"Training Time: {(end-start)/60:.2f} minutes")

# -------- SAVE MODEL BEFORE PLOTS --------
os.makedirs('../outputs', exist_ok=True)
torch.save(model.state_dict(), '../outputs/model.pth')
print("Model saved to ../outputs/model.pth")
# -----------------------------------------

# plots
plt.figure(figsize=(10,7))
plt.plot(train_accuracy, label='train acc')
plt.plot(val_accuracy, label='val acc')
plt.legend()
plt.savefig('../outputs/accuracy.png')
plt.show()

plt.figure(figsize=(10,7))
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()
