from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\maxis\runs\detect\energy_drink_phone_model-4\weights\best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la camara")
    exit()

print("Deteccion en tiempo real iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo la camara")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Deteccion YOLOv8 - Energy Drink & Phone", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
