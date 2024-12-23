from ultralytics import YOLO

# Load model
model = YOLO("yolov8m.pt") 

# Train
model.train(data='data.yaml', 
            epochs=50, 
            imgsz=1024, 
            batch=16, 
            device='2') 