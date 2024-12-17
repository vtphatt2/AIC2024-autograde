from ultralytics import YOLO

# Load model
model = YOLO("yolov5s.pt") 

# Train
model.train(data='data.yaml', epochs=50, imgsz=640, batch=4, device='mps') 
