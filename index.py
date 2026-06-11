from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    name="energy_drink_phone_model"
)

print("Entrenamiento completado")
print("Modelo guardado en: runs/detect/energy_drink_phone_model/weights/best.pt")
