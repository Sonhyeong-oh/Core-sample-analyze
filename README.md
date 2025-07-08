# Core-sample-analyze
시추 코어 이미지 분석

# 1. SAM-YOLO.py
   ![Figure_1](https://github.com/user-attachments/assets/098c056b-7e4e-4444-a26e-0f59639392ac)
   ![Figure_10](https://github.com/user-attachments/assets/eb058e45-ad85-4a06-ae38-839c2fe6abba)

원본 이미지 속 시추 코어 추출을 위한 SAM-YOLO 결합 모델

* SAM : Meta에서 개발한 Visual Transformer 기반 객체탐지 알고리즘
* YOLO : CNN 기반 객체탐지 알고리즘

SAM에서 객체 추출을 하기 위해선 추출하고자 하는 객체의 중심점 또는 영역을 지정이 필요, 

이를 위해 YOLO로 추출하고자 하는 객체에 사각형 경계를 설정한 후, 그 경계를 기준으로 객체를 추출하도록 구성

- 개선사항 : 광물만을 추출하기 위해선 YOLO의 광물 탐지능력이 중요, 광물 이미지 데이터를 확보하여 별도로 학습시킨 모델 구현이 필요

# 참고문헌
* SAM paper : https://arxiv.org/pdf/2304.02643
* Visual Transformer paper : https://arxiv.org/pdf/2010.11929
* YOLO paper : https://arxiv.org/abs/1506.02640
