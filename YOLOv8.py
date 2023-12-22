import os

from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n-cls.pt")  # build a new model from scratch

# Use the model
results = model.train(data="C:\Users\moaya\Desktop\YOLO\traindataset", epochs=200,imgsz=64,patience=300)  # train the model

