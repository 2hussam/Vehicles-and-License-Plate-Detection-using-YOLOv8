from ultralytics import YOLO
# Load a YOLOv8 PyTorch model
model = YOLO('path/to/your/model.pt')

# Export the model
model.export(format='openvino',dynamic=False, half=True)  # creates 'yolov8_openvino_model/'