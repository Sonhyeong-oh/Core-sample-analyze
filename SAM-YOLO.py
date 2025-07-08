# YOLO + SAM 연동 모델

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# === 1. Load YOLOv8 모델 ===
yolo_model = YOLO("yolov8n.pt")  # mineral 전용으로 fine-tuned된 모델 필요

# === 2. Load SAM 모델 ===
sam_checkpoint = "C:/Users/Admin/Downloads/sam_vit_h_4b8939.pth"  # 기학습된 SAM 모델 불러오기
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")
predictor = SamPredictor(sam)

# === 3. 이미지 불러오기 ===
img_path = "C:/Users/Admin/Desktop/사진/시추코어/광물 사진/istockphoto-1291425957-612x612.jpg"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === 4. YOLO로 bounding box 탐지 ===
results = yolo_model(image_rgb)
bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # (x1, y1, x2, y2)

# === 5. SAM에 이미지 등록 ===
predictor.set_image(image_rgb)

# === 6. 각 bounding box에 대해 SAM에 box prompt를 전달하여 마스크 생성 ===
final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for box in bboxes:  # YOLO로 탐지된 bounding box를 SAM의 box 프롬프트로 사용
    masks, scores, _ = predictor.predict(
        box=box,                # SAM에 box prompt로 전달
        multimask_output=False # 가장 confidence 높은 하나의 마스크만 사용
    )
    final_mask = np.logical_or(final_mask, masks[0])  # 마스크 병합
    
# === 7. 마스크로 광물 추출 ===
masked_img = cv2.bitwise_and(image_rgb, image_rgb, mask=final_mask.astype(np.uint8))

# === 8. 시각화 ===
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("YOLO Bounding Boxes")
boxed_img = image_rgb.copy()
for box in bboxes:
    cv2.rectangle(boxed_img, box[:2], box[2:], (255, 0, 0), 2)
plt.imshow(boxed_img)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Extracted Minerals")
plt.imshow(masked_img)
plt.axis("off")

plt.tight_layout()
plt.show()
