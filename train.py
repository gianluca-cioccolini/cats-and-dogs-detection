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


test_img_path = sample_img_path  # puoi cambiare con altre immagini
pred_results = model.predict(source=test_img_path, save=True)

result_img = cv2.imread(pred_results[0].plot())
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.imshow(result_rgb)
plt.axis('off')
plt.title("Predizione YOLOv8")
plt.show()