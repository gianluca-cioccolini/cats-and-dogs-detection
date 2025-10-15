from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

model = YOLO("yolov8n.pt")  # modello nano leggero

dataset_path =  "/workspace/dogs-and-cats-1/data.yaml"

model.train(
    data=dataset_path,
    epochs=100,
    imgsz=640,
)

results = model.val()
print(results)

