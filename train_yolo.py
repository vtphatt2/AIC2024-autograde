from ultralytics import YOLO

# Load model
model = YOLO("yolov5m.pt") 

# Train
model.train(data='data.yaml', epochs=1, imgsz=640, batch=16, device='0') 
