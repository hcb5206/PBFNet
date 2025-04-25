import os
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

input_folder = r"data/PBFNet/images"
output_folder = r"results"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_file_path = os.path.join(input_folder, filename)

        results = model([input_file_path])

        for i, r in enumerate(results):
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])

            output_file_path = os.path.join(output_folder, filename)
            im_rgb.save(output_file_path)

print("检测完成，结果已保存至目标文件夹。")
