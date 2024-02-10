from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="C:\\Users\\zzy\\Desktop\\project\\train-yolov8\\local_env\\config.yaml", epochs=3)  # train the model
