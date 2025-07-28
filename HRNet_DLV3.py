import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
import cv2
from scipy.interpolate import interp1d
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


# =============================================
# HRNet으로 feature map 생성 후 DeepLabV3+로 분석
# =============================================
'''
1. HRNet
고해상도 feature map을 유지한 채로 처리하는 신경망
일반 CNN은 다운샘플링 → 업샘플링 구조를 따르지만, HRNet은 멀티 스케일 feature를 병렬로 유지함

== 추가 설명 ==
=> CNN은 해상도를 줄여나가며 정보 추출 후 다시 해상도를 복원 but 이 과정에서 정보 손실이 발생
=> HRNet은 해상도를 줄이지 않고, 고해상도 feature를 끝까지 유지하면서 다양한 해상도의 정보를 병렬로 처리하고, 지속적으로 서로 교환함.
=> 최종적으로 고해상도 출력 유지

  * 브랜치 : 하나의 신경망 안에서 병렬로 존재하는 경로 또는 구조

2. DeepLabV3+
다양한 스케일의 컨텍스트 정보를 결합해 semantic segmentation(픽셀 단위 객체 분류) 성능 향상
Atrous Convolution (Hole Convolution) 을 활용하여 downsampling 없이 receptive field 확장
ASPP (Atrous Spatial Pyramid Pooling): 서로 다른 dilation rate를 가진 여러 Conv layer를 병렬로 배치
Decoder: DeepLabV3+에서는 decoder 모듈 추가로 경계선 복원 강화
'''

class HRNet_DeepLabV3(nn.Module):
    def __init__(self, hrnet_variant='hrnet_w18', out_dim=3):
        super().__init__()

        # 1. HRNet 백본 (features_only=True → intermediate feature map 추출)
        self.hrnet = timm.create_model(
            hrnet_variant,
            pretrained=True,
            features_only=True  # 마지막 feature map을 리스트로 반환
        )

        # 2. DeepLabV3+ Head (ASPP + Decoder)
        # HRNet 마지막 채널 수 확인
        in_channels = self.hrnet.feature_info[-1]['num_chs']  # 예: 720 for hrnet_w18
        self.aspp_head = DeepLabHead(in_channels, 256)

        # 3. 회귀 헤드
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # GAP
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)  # 예: 3개 물성값
        )

    def forward(self, x):
        # HRNet backbone → feature map list
        features = self.hrnet(x)
        feat = features[-1]  # 가장 마지막 feature map 사용

        # DeepLabV3+ ASPP Head
        x = self.aspp_head(feat)

        # 회귀
        out = self.regressor(x)
        return out

# ==================================    
# csv 데이터 전처리
# ==================================
'''
csv 파일을 확인해보면 공란(결측치)가 있음
이는 데이터 변환 시 NaN으로 처리되어 에러를 발생시킴
따라서 데이터 전처리를 통해 해결해야 함

사용 방법 : 다항식 보간 (ploynomial Interpolation)
* 보간 : 결측치나 중간의 빠진 데이터를 채워 넣는 기법
* 다항식 보간 방법 :
  주어진 데이터 포인트들을 통과하는 다항식(polynomial function)을 만들어, 그 다항식을 기반으로 중간 값을 추정하는 방법

=> 데이터 그래프의 경향을 부드럽게 추종하는 값을 채워넣음
'''

# 입력 및 출력 디렉토리
input_dir = "C:/Users/Admin/Desktop/2nd core sample dataset/log data"  # csv 데이터 폴더 지정
output_dir = "C:/Users/Admin/Desktop/2nd core sample dataset/Interpooled_data" # csv 보간 결과를 저장할 폴더 지정
os.makedirs(output_dir, exist_ok=True)

# 보간 대상 열 및 x축
columns_to_interpolate = ["pwave_vel", "density", "mag_sus"]
x_axis = "sb_depth_cm"

# 모든 CSV 처리
csv_files = glob.glob(f"{input_dir}/*.csv")
print(f"총 {len(csv_files)}개의 파일 처리 중...")

for file_path in csv_files:
    filename = os.path.basename(file_path).replace(".csv", "")
    print(f"📄 처리 중: {filename}.csv")

    # 1. 파일 불러오기
    df = pd.read_csv(file_path)
    original_df = df.copy()

    # 2. 다항식 보간
    for col in columns_to_interpolate:
        if df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(method="polynomial", order=2)
            df[col] = df[col].ffill().bfill()

    # 3. 표준화 (원본/보간 데이터 모두)
    scaler = StandardScaler()
    original_df[columns_to_interpolate] = scaler.fit_transform(original_df[columns_to_interpolate])
    df[columns_to_interpolate] = scaler.fit_transform(df[columns_to_interpolate])

    # 4. 시각화 (옵션)
    # for col in columns_to_interpolate:
    #     mv_before = original_df[col].isna().sum()
    #     mv_after = df[col].isna().sum()

    #     fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    #     axs[0].plot(original_df[x_axis], original_df[col], marker='o', color='red')
    #     axs[0].set_title(f"[Before] {col} (NaNs: {mv_before})")
    #     axs[0].set_xlabel("Depth (cm)")
    #     axs[0].set_ylabel(col)
    #     axs[0].grid(True)

    #     axs[1].plot(df[x_axis], df[col], marker='o', color='blue')
    #     axs[1].set_title(f"[After] {col} (NaNs: {mv_after})")
    #     axs[1].set_xlabel("Depth (cm)")
    #     axs[1].grid(True)

    #     plt.tight_layout()
    #     plt.show()

    # 5. 저장
    save_path = os.path.join(output_dir, f"{filename}_interpooled.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ 저장 완료: {save_path}\n")


# ========== 설정 ==========
image_dir = "C:/Users/Admin/Desktop/2nd core sample dataset/raw image" # 원본 이미지 데이터 폴더 지정
csv_dir = "C:/Users/Admin/Desktop/2nd core sample dataset/Interpooled_data" # 보간된 csv 데이터 폴더 지정
output_dir = "C:/Users/Admin/Desktop/2nd core sample dataset/matched_data" # 이미지와 csv 데이터를 연결한 데이터를 저장할 폴더 지정
os.makedirs(output_dir, exist_ok=True)

pixels_per_cm = 200  # 1cm = 200px (xml 파일 확인)

# ========== 이미지와 csv 파일 구분을 위한 확장자 이름 지정 함수 ==========
def sort_by_section(img_list):
    def extract_number(name):
        return int(name.split("_")[-1].replace(".tif", ""))
    return sorted(img_list, key=extract_number)

def extract_core_id(name):
    parts = name.replace(".tif", "").replace(".csv", "").split("_")
    return "_".join(parts[2:4])  # "FA_GC01"

# ========== 파일 목록 정리 ==========
# 1. 이미지
all_images = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
image_map = {}
for img in all_images:
    core_id = extract_core_id(img)
    image_map.setdefault(core_id, []).append(img)

# 2. CSV
all_csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
csv_map = {}
for csv in all_csvs:
    core_id = extract_core_id(csv)
    csv_map[core_id] = csv

# 3. 공통 core_id만 추출
image_core_ids = set(image_map.keys())
csv_core_ids = set(csv_map.keys())
common_ids = image_core_ids & csv_core_ids

# 4. 누락된 데이터 확인
image_only = image_core_ids - csv_core_ids
csv_only = csv_core_ids - image_core_ids

print("❌ 이미지만 있고 CSV 없는 core_id:", sorted(image_only))
print("❌ CSV만 있고 이미지 없는 core_id:", sorted(csv_only))
print("✅ 처리 가능한 core_id 수:", len(common_ids))

# 5. 이미지와 csv 파일에서 명시된 길이를 비교하여 전처리
for core_id in tqdm(sorted(common_ids), desc="🔁 Core 샘플 처리"):
    # ----------------------
    # 1. 이미지 병합
    # ----------------------
    merged_image = None
    sorted_imgs = sort_by_section(image_map[core_id])
    for fname in sorted_imgs:
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        merged_image = img if merged_image is None else np.vstack((merged_image, img))

    # ----------------------
    # 2. CSV 로드
    # ----------------------
    csv_path = os.path.join(csv_dir, csv_map[core_id])
    df = pd.read_csv(csv_path)
    target_cols = ["density", "pwave_vel", "mag_sus"]
    depth_col = "sb_depth_cm"

    # ----------------------
    # 3. 이미지 슬라이스 수 계산
    # ----------------------
    image_height_px = merged_image.shape[0]
    num_slices = image_height_px // pixels_per_cm
    image_depths = np.arange(num_slices)

    # ----------------------
    # 4. 보간 또는 절단
    # ----------------------
    if num_slices > len(df):
        print(f"🟢 {core_id}: 이미지가 더 길어서 보간 수행")
        df_interp = pd.DataFrame({depth_col: image_depths})
        for col in target_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].interpolate(method="polynomial", order=2)
                df[col] = df[col].ffill().bfill()
            f = interp1d(df[depth_col], df[col], kind='linear', fill_value="extrapolate")
            df_interp[col] = f(image_depths)
        df = df_interp

    elif num_slices < len(df):
        print(f"🟡 {core_id}: 이미지가 더 짧아서 자름")
        df = df.iloc[:num_slices].copy()

    df = df.dropna(subset=target_cols).reset_index(drop=True)

    # ----------------------
    # 5. 이미지 슬라이스 저장
    # ----------------------
    sliced_image_paths = []
    core_output_dir = os.path.join(output_dir, core_id)
    os.makedirs(core_output_dir, exist_ok=True)

    for i in range(len(df)):  # 보간 후 길이 기준으로 자름
        y0 = i * pixels_per_cm
        y1 = (i + 1) * pixels_per_cm
        crop = merged_image[y0:y1, :]
        out_path = os.path.join(core_output_dir, f"{core_id}_slice_{i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        sliced_image_paths.append(out_path)

    # ----------------------
    # 6. 최종 CSV 저장
    # ----------------------
    df["image_path"] = sliced_image_paths
    save_csv_path = os.path.join(core_output_dir, f"{core_id}_matched.csv")
    df.to_csv(save_csv_path, index=False)
    print(f"✅ 저장 완료: {core_id} → {save_csv_path}")

# 데이터 선언 클래스
class CoreImageDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)  # 💡 DataFrame 직접 받도록 수정
        self.image_paths = self.df["image_path"].values
        self.targets = self.df[["density", "pwave_vel", "mag_sus"]].values.astype("float32")
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(self.targets[idx])
        return image, target

# -----------------------------
# 설정
# -----------------------------
batch_size = 32 # 배치 사이즈 : 한 번의 학습동안 입력될 데이터의 개수
epochs = 50 # 에포크 : 몇 번의 학습을 반복할지 설정
lr = 1e-4 # 학습률 : 파라미터 업데이트의 속도 설정
                  # 너무 크면 발산하여 적절한 파라미터를 찾지 못함 & 너무 작으면 학습 속도가 느려짐
out_dim = 3  # 회귀 대상 수 (밀도, 속도, 자성 등 물성 column의 개수를 지정)

# gpu가 있으면 gpu 사용, 없으면 cpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 데이터 불러오기
# -----------------------------
matched_root = "C:/Users/Admin/Desktop/2nd core sample dataset/matched_data"
all_core_ids = sorted(os.listdir(matched_root))

train_dfs, val_dfs, test_dfs = [], [], []

for core_id in all_core_ids:
    matched_csv = os.path.join(matched_root, core_id, f"{core_id}_matched.csv")
    if not os.path.exists(matched_csv):
        continue

    df = pd.read_csv(matched_csv)

    # core_id 열 추가
    df["core_id"] = core_id

    # 슬라이스 단위 분할
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # val 15%, test 15%

    train_dfs.append(train_df)
    val_dfs.append(val_df)
    test_dfs.append(test_df)

# 전체 데이터프레임 통합
train_df = pd.concat(train_dfs, ignore_index=True)
val_df = pd.concat(val_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"Train samples: {len(train_df)}")
print(f"Val samples  : {len(val_df)}")
print(f"Test samples : {len(test_df)}")

# 4. Dataset / DataLoader 생성
train_ds = CoreImageDataset(train_df)
val_ds = CoreImageDataset(val_df)
test_ds = CoreImageDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# -----------------------------
# 모델 정의
# -----------------------------
model = HRNet_DeepLabV3(hrnet_variant='hrnet_w18', out_dim=out_dim)
model = model.to(device)

# -----------------------------
# 손실 함수 및 옵티마이저
# -----------------------------
loss_fn = nn.MSELoss() # 평균제곱오차를 기준으로 Loss 계산
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# 학습 루프
# -----------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)

        if torch.isnan(preds).any():
            print("NaN in preds! Skipping batch.")
            continue

        loss = loss_fn(preds, targets)

        if torch.isnan(loss):
            print("NaN in loss! Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # 결과 출력
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    # Loss : 모델의 예측값과 실제값 사이의 오차를 수치로 나타낸 것 (작을수록 좋음)
    # RMSE : MSE의 제곱근을 취한 값 (= 현재 Loss가 MSELoss이기 때문에 Loss의 제곱근 값임) (작을수록 좋음)
    # R square(결정계수) : 모델이 실제 데이터를 얼마나 잘 설명하는지를 나타내는 지표 (클수록 좋음)
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | RMSE: {rmse:.3f} | R²: {r2:.3f}\n")

# 학습 완료된 모델 저장
torch.save(model.state_dict(), "hrnet_dlv3_regression.pth")

# -----------------------------
# 테스트 데이터 예측 및 평가
# -----------------------------
model = HRNet_DeepLabV3(hrnet_variant='hrnet_w18', out_dim=3)  # 아키텍처 반드시 동일하게
model.load_state_dict(torch.load("hrnet_dlv3_regression.pth"), strict = False)
model = model.to(device)
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images).cpu().numpy()
        y_pred.append(outputs)
        y_true.append(targets.numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# -----------------------------
# 정량 지표 계산
# -----------------------------
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("========================")
print("📊 테스트 결과 (정량 지표)")
print(f"🔹 MSE : {mse:.4f}")
print(f"🔹 RMSE: {rmse:.4f}")
print(f"🔹 R²  : {r2:.4f}")
print("========================")

# -----------------------------
# 실제값 vs 예측값 그래프
# -----------------------------
feature_names = ["Density", "P-wave Velocity", "Magnetic Susceptibility"]
num_features = y_true.shape[1]  # 보통 3개

plt.figure(figsize=(18, 5))

for i in range(num_features):
    plt.subplot(1, 3, i + 1)
    plt.plot(y_true[:, i], label="True", color="blue", marker="o")
    plt.plot(y_pred[:, i], label="Pred", color="red", linestyle="--", marker="x")
    plt.title(f"{feature_names[i]} (per Slice)")
    plt.xlabel("Slice Index (1cm units)")
    plt.ylabel(feature_names[i])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()