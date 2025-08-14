# Core-sample-analyze
Core sample image analysis

# 1. SAM-YOLO.py
   ![Figure_1](https://github.com/user-attachments/assets/098c056b-7e4e-4444-a26e-0f59639392ac)
   ![Figure_10](https://github.com/user-attachments/assets/eb058e45-ad85-4a06-ae38-839c2fe6abba)

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

# 3. HRNet+DLV3.py

<img width="1758" height="704" alt="image" src="https://github.com/user-attachments/assets/ce525b9a-758f-43d3-b26b-06e197277c3b" />

A model that predicts density, magnetic susceptibility, and P-wave velocity from core sample images.

- Improvements

Add hyperparameter tuning code

Extend training up to 100 epochs for testing

Modify visualization method

# 4. SLIC_SAM.py

<img width="796" height="740" alt="05" src="https://github.com/user-attachments/assets/fc1602a9-eac0-4834-be8c-c532bcb8247b" />

Image Segmentation Using SLIC(Simple Linear Iterative Clustering) and SAM(Segment Anything Model)

SLIC (Simple Linear Iterative Clustering) generates initial superpixels based on the image's color and location. The center points of these generated superpixels are then used as prompts for SAM (Segment Anything Model) to create more detailed and precise final segmentation masks.

# 5. composition_pred.py

<img width="941" height="785" alt="2019_643_fa_jpc40_01_left50_cropped_viz" src="https://github.com/user-attachments/assets/8ad05233-36ae-4cbc-8e64-2bc7ce1af86b" />

SLIC (Superpixel Linear Iterative Clustering) is a variation of K-means clustering that operates in a color and spatial domain. 
It searches only the local neighborhood of each cluster center to group pixels with high color and spatial similarity. 

The areas identified by SLIC as potential different minerals are then fed into the Segment Anything Model (SAM) algorithm. Using a Vision Transformer, SAM outputs multiple candidate segmentation results. The most confident segmentation result is then selected from these candidates, which is designed to improve the precision of the area segmentation. 

Finally, the extracted areas, predicted physical property values, and physical property variation patterns of each component are used as inputs for Non-Negative Least Squares (NNLS) regression to calculate the compositional percentage for each area.

# References
* SAM : https://arxiv.org/pdf/2304.02643
* Visual Transformer : https://arxiv.org/pdf/2010.11929
* YOLO : https://arxiv.org/abs/1506.02640
* HRNet : https://arxiv.org/abs/1908.07919
* DeepLabV3+ : https://arxiv.org/abs/1802.02611
