from ultralytics import YOLO

model = YOLO("yolo11m-obb.pt") # Pretrained model on DOTAv1

model.train(
    data="PLE/A_dataset/dataset",
    pretrained=True,
    imgsz=512, 
    epochs=5000, # who knows how long it will take...
    mosaic=0.5,
    degrees=180,
    translate=0.0,
    scale=0.1,
    flipud=0.5,
    fliplr=0.5,
)