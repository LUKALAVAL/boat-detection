from ultralytics import YOLO

model = YOLO("PLA/weights.pt")

model.val(
    data="PLA/dataset/dataset.yaml",
    imgsz=512,
    conf=0.370,
    iou=0.3,
    project='PLA/A_model/runs/obb',
    split='test',
    name='test',
)