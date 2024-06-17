from ultralytics import YOLO

model = YOLO("../Vechles Tracking and license plate detection git version/Vehicles-and-License-Plate-Detection-using-YOLOv8/licensePlate_openvino_model_yolov8n")
model.predict(source ="E:\Vechles Tracking and license plate detection git version\Vehicles-and-License-Plate-Detection-using-YOLOv8\Vehcles_4k.mp4", show = True, conf = 0.3, save=True)