# Core-sample-analyze
Core sample image analysis

# 1. SAM-YOLO.py
   ![Figure_1](https://github.com/user-attachments/assets/098c056b-7e4e-4444-a26e-0f59639392ac)
   ![Figure_10](https://github.com/user-attachments/assets/eb058e45-ad85-4a06-ae38-839c2fe6abba)

원본 이미지 속 시추 코어 추출을 위한 SAM-YOLO 결합 모델

A combined SAM-YOLO model for extracting core samples from original images.

- SAM: A Visual Transformer-based object segmentation algorithm developed by Meta

- YOLO: A CNN-based object detection algorithm

In order to extract objects using SAM, it is necessary to specify the center point or region of the target object.

To achieve this, YOLO is used to define bounding boxes for the target object, and then SAM extracts the object based on that region.

- Improvements: For extracting only minerals, the mineral detection capability of YOLO is crucial. It is necessary to train a custom model using a dedicated mineral image dataset.

# 2. SAM_D&D.py

![Figure_2](https://github.com/user-attachments/assets/5641c6ec-6b12-4749-aa91-9a40e07d8856)
![Figure_3](https://github.com/user-attachments/assets/893749ea-17e6-40ee-bd5d-42b1a58350d1)
![Figure_4](https://github.com/user-attachments/assets/f701825d-3a12-4359-8033-feeabdaa2f7c)

A standalone SAM script for manually specifying regions when YOLO's object detection is unreliable.
If a box is manually defined, SAM detects objects within that region.

# References
* SAM paper : https://arxiv.org/pdf/2304.02643
* Visual Transformer paper : https://arxiv.org/pdf/2010.11929
* YOLO paper : https://arxiv.org/abs/1506.02640

# 3. HRNet+DLV3.py

<img width="1727" height="683" alt="image" src="https://github.com/user-attachments/assets/6399024b-9afe-4c40-8506-c295c067cc9d" />

A model that predicts density, magnetic susceptibility, and P-wave velocity from core sample images.

- Improvements

Add hyperparameter tuning code

Extend training up to 100 epochs for testing

Modify visualization method

시추코어 이미지를 통한 밀도, 자기감수율, P파 속도 예측 모델

* 개선 사항
  - 파라미터 튜닝 코드 추가
  - epoch를 100까지 늘려서 테스트
  - 시각화 방식 변경
