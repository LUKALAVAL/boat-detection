from ultralytics import YOLO

model = YOLO("PLE/weights.pt")

model.predict(
    source="PLE/A_dataset/dataset/images/test",
    imgsz=512,
    conf=0.001,
    iou=0.3,
    save_txt=True,
    save_conf=True,
)