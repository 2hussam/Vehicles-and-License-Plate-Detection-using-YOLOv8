from ultralytics import YOLO

model_path = "path to your model"
source = "path to your imag or video or put 0 to open your webcam"

model = YOLO(model_path)
model.predict(source = source, show = True, conf = 0.3, save= True)