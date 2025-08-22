from ultralytics import YOLO

model = YOLO("yolo11m-obb.pt")

model.train(
    data="PLA/dataset/dataset.yaml",
    pretrained=True,
    imgsz=512, 
    epochs=1800,
    mosaic=0.7,
    degrees=180,
    translate=0.1,
    flipud=0.5,
    fliplr=0.5,
)