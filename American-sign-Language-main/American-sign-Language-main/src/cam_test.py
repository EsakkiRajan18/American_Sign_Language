'''
USAGE:
python cam_test.py
'''

import torch
import joblib
import numpy as np
import cv2
import time
import cnn_models
import matplotlib.pyplot as plt
from collections import deque

# load label binarizer
lb = joblib.load('../outputs/lb.pkl')

model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('../outputs/model.pth'))
model.eval()
print('Model loaded')

# smoothing buffer
pred_buffer = deque(maxlen=7)

def hand_area(img):
    # bigger region for fingers
    hand = img[80:360, 80:360]
    hand = cv2.resize(hand, (224, 224))
    return hand

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error while trying to open camera.')
    exit()

plt.ion()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (80, 80), (360, 360), (20, 34, 255), 2)
    hand = hand_area(frame)

    # normalize like training
    image = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    pred_label = lb.classes_[preds.item()]

    # smooth predictions
    pred_buffer.append(pred_label)
    pred_label = max(set(pred_buffer), key=pred_buffer.count)

    # draw text
    cv2.putText(frame, pred_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # matplotlib live view
    plt.clf()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {pred_label}")
    plt.axis("off")
    plt.pause(0.001)

    time.sleep(0.02)

plt.close()
cap.release()
