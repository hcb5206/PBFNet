from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.val(data="data.yaml", batch=1, device="0")

print(f'P: {results.box.mp:.4f}')
print(f'R: {results.box.mp:.4f}')
print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
