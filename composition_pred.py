import os, re, glob, cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch.nn as nn
import torchvision.transforms as T
from scipy.optimize import nnls
from difflib import SequenceMatcher
from PIL import Image, UnidentifiedImageError, ImageFile  # ← ImageFile 추가
Image.MAX_IMAGE_PIXELS = None            # 거대 이미지 경고/오류 비활성화
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 손상/잘린 이미지도 로드 시도
import matplotlib.cm as cm
import traceback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from pathlib import Path

def load_image_rgb_safely(img_path: str):
    """
    1) OpenCV로 먼저 읽고(BGR→RGB), 실패시 2) Pillow로 폴백.
    멀티프레임 TIFF는 첫 프레임으로 강제.
    문제 발생 시 None 반환하고 콘솔에 원인 출력.
    """
    # 1) OpenCV 시도 (대용량/특수 TIF에서 더 튼튼한 경우 많음)
    try:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            return pil
    except Exception as e:
        print(f"[WARN] cv2.imread failed for: {img_path} -> {type(e).__name__}: {e}")

    # 2) Pillow 폴백
    try:
        with Image.open(img_path) as im:
            # 멀티프레임 TIFF 대비: 첫 프레임으로 이동
            if getattr(im, "n_frames", 1) > 1:
                try:
                    im.seek(0)
                except Exception:
                    pass
            im = im.convert("RGB")
            return im
    except Exception as e:
        print(f"[ERR] PIL open/convert failed for: {img_path}\n      -> {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)
        return None

# =========================
# 0) 모델 (HRNet+DeepLabV3)
# =========================
class HRNet_DeepLabV3(nn.Module):
    # HRNet + DeepLabV3 head + GAP regressor
    def __init__(self, hrnet_variant='hrnet_w18', hidden_dim=128, out_dim=3):
        super().__init__()
        import timm
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead

        self.hrnet = timm.create_model(hrnet_variant, pretrained=True, features_only=True)
        in_channels = self.hrnet.feature_info[-1]['num_chs']
        self.aspp_head = DeepLabHead(in_channels, 256)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)  # (rho, Vp, chi)
        )
    
    def forward(self, x):
        feat = self.hrnet(x)[-1]
        x = self.aspp_head(feat)
        out = self.regressor(x)
        return out

    def forward_features(self, x):
        feat = self.hrnet(x)[-1]
        x = self.aspp_head(feat)      # (B,256,Hf,Wf)
        return x

    def regress_from_vec(self, vec_256):
        fc_head = nn.Sequential(*list(self.regressor)[2:])  # GAP/Flatten 제외
        return fc_head(vec_256)  # (B,3)

# =========================
# 1) NNLS 설정 + 저장 유틸
# =========================
# NNLS 설정: 탄산염, 규산염, 점토, 산화철의 조성 비율을 예측하기 위한 행렬
A = np.array([
    # 탄산염  # 규산염  # 점토  # 산화철
    [ 1.00,  0.30, -0.80,  3.00],   # 밀도
    [ 0.80,  0.30, -0.80,  0.80],   # P파 속도
    [-0.20, -0.20,  0.20,  3.00],   # 자화율
], dtype=float)
W = np.diag([1.0, 1.0, 0.3])
lam = 0.2
A_aug = np.vstack([W @ A, lam * np.ones((1, A.shape[1]))])
labels4 = np.array(["carbonate","silicate","clay","fe_oxides"])

# NNLS를 위한 함수 정의
def nnls_frac(y_z):
    y_aug = np.concatenate([W @ y_z, [lam]])
    f, _ = nnls(A_aug, y_aug)
    s = f.sum()
    return f/s if s > 0 else f

# 안전하게 DataFrame을 CSV로 저장하는 유틸
def safe_to_csv(df: pd.DataFrame, out_path: str, label: str):
    out_path = str(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    try:
        df.to_csv(out_path, index=False, encoding="utf-8")
        if os.path.exists(out_path):
            print(f"[OK] {label} saved: {out_path}")
        else:
            print(f"[ERR] {label} not found after save: {out_path}")
    except Exception as e:
        print(f"[ERR] Failed to save {label}: {out_path}\n      -> {type(e).__name__}: {e}")

# NNLS 결과 컬럼 정의
NNLS_COLS = [
    "seg_id",
    "carbonate","silicate","clay","fe_oxides",
    "dominant","confidence","residual_rms","seg_center_row"
]

# NNLS-only 저장 유틸
def save_nnls_only(df: pd.DataFrame, out_path: str, label: str):
    cols = [c for c in NNLS_COLS if c in df.columns]
    if not cols:
        print(f"[WARN] {label}: NNLS columns not found -> skip ({NNLS_COLS})")
        return
    df_nnls = df[cols].copy()
    safe_to_csv(df_nnls, out_path, f"{label} (NNLS-only)")

# =========================
# 2) 파일명 파서/수집
# =========================
# 대소문자 무시 + 반환은 대문자 통일
core_id_pat = re.compile(r"(fa_[a-z]+?\d+)", re.IGNORECASE)
def extract_core_id(filename:str):
    m = core_id_pat.search(filename)
    return m.group(1).upper() if m else "UNKNOWN"

# 섹션: 01, 1, sec01, section1 등 대응
sec_pat = re.compile(r"(?:^|[_\-])(sec|section)?\s*0*?(\d{1,3})(?:[_\-]|$)", re.IGNORECASE)
def extract_section_idx(filename:str):
    m = sec_pat.search(filename)
    return int(m.group(2)) if m else 999

# 재귀적으로 이미지 파일 수집
def collect_images_recursive(image_dir, exts=('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp')):
    patterns = []
    for e in exts:
        e = e.lower()
        patterns.append(f"**/*{e}")
        patterns.append(f"**/*{e.upper()}")
    files = []
    for p in patterns:
        files.extend(glob.glob(str(Path(image_dir) / p), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    # 윈도우 중복 제거
    files = list(dict.fromkeys([os.path.normcase(os.path.normpath(f)) for f in files]))
    return sorted(files)

# =========================
# 3) 마스크 찾기 (매칭 + 스코어링)
# =========================
# 매칭을 위한 스코어링 함수 (기본 이름과 후보 경로 비교)
def _score_candidate(base:str, cand_path:str):
    cand_name = os.path.splitext(os.path.basename(cand_path))[0]
    s1 = SequenceMatcher(None, base, cand_name).ratio()
    core_base = extract_core_id(base)
    sec_base  = extract_section_idx(base)
    bonus = 0.0
    if core_base in cand_name.upper(): bonus += 0.1
    if f"{sec_base:02d}" in cand_name or str(sec_base) in cand_name: bonus += 0.1
    return s1 + bonus

# 영역 마스크 파일 찾기 (정확 일치 + 느슨 매칭)
def find_mask_for_image(base:str, pseudo_label_dir:str, verbose=False):
    # 0) 정확 일치
    exact = os.path.join(pseudo_label_dir, f"{base}_merged.npy")
    if os.path.exists(exact):
        if verbose: print(f"[DBG] exact match -> {exact}")
        return exact

    core_id = extract_core_id(base)
    sec_idx = extract_section_idx(base)

    patterns = [
        f"*{core_id}*{sec_idx:02d}*merged*.npy",
        f"*{core_id}*{sec_idx}*merged*.npy",
        f"*{core_id}*merged*.npy",
        # 필요하면 아래 두 줄도 열어주세요 (merged 접미사가 없을 때)
        # f"*{core_id}*{sec_idx:02d}*.npy",
        # f"*{core_id}*{sec_idx}*.npy",
    ]
    cand_all = []
    for pat in patterns:
        cand_all += glob.glob(os.path.join(pseudo_label_dir, pat))

    if not cand_all:
        # 토큰 기반 느슨 매칭
        tokens = re.split(r'[_\-]+', base)
        if core_id not in tokens:
            tokens.append(core_id)
        pat = "*" + "*".join(tokens[:4]) + "*merged*.npy"
        cand_all = glob.glob(os.path.join(pseudo_label_dir, pat))

    if not cand_all:
        if verbose: print(f"[DBG] no candidates for base='{base}'")
        return None

    cand_all = list(set(cand_all))
    cand_all.sort(key=lambda p: _score_candidate(base, p), reverse=True)
    if verbose:
        print("[DBG] candidates ranked:")
        for c in cand_all[:5]:
            print("   ", _score_candidate(base, c), "->", c)
    return cand_all[0]

# =========================
# 4) 한 이미지 처리
# =========================
# GPU/CPU 안전하게 이동하는 유틸
def _to_device(x, device):
    try:
        return x.to(device)
    except Exception as e:
        print(f"[GPU->CPU] tensor move failed on '{device}': {type(e).__name__}: {e}")
        return x.cpu()

# forward_features를 안전하게 호출하는 유틸
def _safe_forward_features(model, x, device):
    # 1차: 지정 device
    try:
        return model.forward_features(_to_device(x, device))
    except RuntimeError as e:
        if "CUDA" in str(e).upper():
            print(f"[GPU->CPU] forward_features CUDA error: {e}\n          Falling back to CPU for this image.")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model_cpu = model.to("cpu")
            x_cpu = x.cpu()
            with torch.no_grad():
                return model_cpu.forward_features(x_cpu)
        raise

# 2차: CPU로 폴백
def segment_props_and_fractions(model, img_path, mask_npy_path, device="cuda", min_pixels=5):
    transform = T.Compose([
        T.Resize((256,256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    try:
        print(f"[STEP] open image -> {img_path}")
        pil = load_image_rgb_safely(img_path)
        if pil is None:
            print(f"[WARN] skip image due to load failure: {img_path}")
            return pd.DataFrame()
        
        print(f"[STEP] image size = {pil.size}")
        x = transform(pil)[None]   # ← device 이동은 _safe_forward_features가 처리
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        print(f"[ERR] PIL open failed: {type(e).__name__}: {e}")
        return pd.DataFrame()

    try:
        x = transform(pil)[None]
        print(f"[STEP] load mask -> {mask_npy_path}")
        mask_full = np.load(mask_npy_path)
    except Exception as e:
        print(f"[ERR] preprocess failed: {type(e).__name__}: {e}")
        return pd.DataFrame()

    if mask_full.ndim != 2:
        print(f"[WARN] mask ndim!=2: shape={mask_full.shape}")
        return pd.DataFrame()

    mask_256 = cv2.resize(mask_full, (256,256), interpolation=cv2.INTER_NEAREST)

    model.eval()
    with torch.no_grad():
        try:
            print("[STEP] forward_features...")
            fmap = _safe_forward_features(model, x, device)   # ★ 폴백 사용
        except Exception as e:
            print(f"[ERR] forward_features failed (both GPU/CPU): {type(e).__name__}: {e}")
            return pd.DataFrame()

        _, C, Hf, Wf = fmap.shape
        print(f"[STEP] fmap shape = (1,{C},{Hf},{Wf})")
        mask_f = cv2.resize(mask_256, (Wf,Hf), interpolation=cv2.INTER_NEAREST)
        seg_ids = [sid for sid in np.unique(mask_f) if sid != 0]
        print(f"[STEP] seg_ids = {seg_ids[:10]} (total {len(seg_ids)})")

        Ys, rows = [], []
        for sid in seg_ids:
            m = (mask_f == sid).astype(np.float32)
            if m.sum() < min_pixels:
                continue
            m_t = torch.from_numpy(m)[None,None]
            area = m_t.sum().clamp_min(1.0)
            # fmap와 동일 디바이스로
            m_t = m_t.to(fmap.device)
            pooled = (fmap * m_t).sum(dim=[2,3]) / area
            try:
                y_hat = model.regress_from_vec(pooled).squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"[ERR] regress_from_vec failed (seg {sid}): {type(e).__name__}: {e}")
                continue

            ys, xs = np.where(mask_f == sid)
            center_row = float(ys.mean()) if len(ys) else np.nan
            Ys.append(y_hat)
            rows.append({"seg_id": int(sid), "rho": y_hat[0], "Vp": y_hat[1], "chi": y_hat[2], "seg_center_row": center_row})

    if not rows:
        print(f"[WARN] no rows after pooling (min_pixels={min_pixels})")
        return pd.DataFrame(columns=["seg_id","rho","Vp","chi","carbonate","silicate","clay","fe_oxides","dominant","confidence","residual_rms","seg_center_row"])

    print("[STEP] NNLS...")
    Y = np.vstack(Ys)
    mu, sd = Y.mean(0), Y.std(0) + 1e-9
    Z = (Y - mu) / sd
    try:
        F = np.vstack([nnls_frac(z) for z in Z])
    except Exception as e:
        print(f"[ERR] NNLS failed: {type(e).__name__}: {e}")
        return pd.DataFrame()

    top1 = F.argmax(1); top2 = np.argsort(F,1)[:,-2]
    conf = F[np.arange(F.shape[0]), top1] - F[np.arange(F.shape[0]), top2]
    resid = np.linalg.norm((W @ (A @ F.T) - W @ Z.T).T, axis=1) / np.sqrt(3)

    df = pd.DataFrame(rows)
    df[["carbonate","silicate","clay","fe_oxides"]] = F
    df["dominant"] = np.array(["carbonate","silicate","clay","fe_oxides"])[top1]
    df["confidence"] = conf
    df["residual_rms"] = resid
    print(f"[STEP] done -> {len(df)} segments")
    return df

# ============
# 시각화 설정
# ============
# RGB 변환 유틸 (OpenCV BGR → Matplotlib RGB)
def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 시각화 함수: 세그 라벨을 label 0, 1, 2…로 표기하고 각 라벨 조성을 박스로 표시
def visualize_compositions(
    img_path: str,
    mask_npy_path: str,
    df_seg: pd.DataFrame,
    out_png: str,
    topk: int = 3,                 # 말풍선으로 자세히 적을 라벨 수(면적 큰 순)
    overlay_all_ids: bool = True,  # 모든 라벨의 번호를 세그 위에 숫자로 찍기
    max_legend: int = 12,          # 범례에 보여줄 라벨 수 제한(많으면 생략)
):
    """세그 라벨을 label 0, 1, 2…로 표기하고 각 라벨 조성을 박스로 표시."""
    # 1) 원본/마스크 로드
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] visualize: cannot open image: {img_path}")
        return
    img_rgb = _rgb(img_bgr)
    H, W = img_rgb.shape[:2]

    mask_full = np.load(mask_npy_path)
    if mask_full.shape[:2] != (H, W):
        mask_full = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

    # 2) 라벨 목록(세그 ID)과 0부터 시작하는 연속 인덱스 매핑
    if df_seg.empty:
        print(f"[WARN] visualize: df_seg empty -> {img_path}")
        return
    seg_ids = sorted(int(s) for s in df_seg["seg_id"].unique())
    id2idx = {sid: i for i, sid in enumerate(seg_ids)}  # 실제 seg_id -> label 0..K-1

    # 3) 라벨별 색상(고정 팔레트)
    cmap = cm.get_cmap('tab20', max(20, len(seg_ids)))
    def color_for_idx(idx):
        r, g, b, _ = cmap(idx % cmap.N)
        return (int(255*r), int(255*g), int(255*b))

    seg_vis = np.zeros((H, W, 3), dtype=np.uint8)
    for sid in seg_ids:
        seg_vis[mask_full == sid] = color_for_idx(id2idx[sid])

    # 4) 면적 큰 순으로 topk 라벨 선택 (말풍선 대상)
    areas = [(sid, int((mask_full == sid).sum())) for sid in seg_ids]
    areas.sort(key=lambda x: -x[1])
    pick_ids = [sid for sid, _ in areas[:max(0, topk)]]

    # 5) 플롯
    fig = plt.figure(figsize=(10, 8), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.04)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.imshow(img_rgb); ax0.set_title(f"Original Image: {Path(img_path).name}", fontsize=10); ax0.axis("off")
    ax1.imshow(seg_vis); ax1.set_title(f"Segmentation ({len(seg_ids)} labels)", fontsize=10); ax1.axis("off")

    # 6) 모든 라벨의 번호를 세그 중앙에 찍기(원하면 끄기)
    if overlay_all_ids:
        for sid in seg_ids:
            ys, xs = np.where(mask_full == sid)
            if len(xs) == 0: continue
            cx, cy = float(xs.mean()), float(ys.mean())
            ax1.text(cx, cy, str(id2idx[sid]),
                     fontsize=7, ha='center', va='center',
                     bbox=dict(fc='white', ec='none', alpha=0.6, pad=0.2))

    # 7) 말풍선 박스(면적 큰 topk 라벨)
    for sid in pick_ids:
        row = df_seg[df_seg["seg_id"] == sid].iloc[0]
        ys, xs = np.where(mask_full == sid)
        if len(xs) == 0: continue
        cx, cy = float(xs.mean()), float(ys.mean())

        carb = float(row["carbonate"]) * 100.0
        sili = float(row["silicate"])  * 100.0
        clay = float(row["clay"])      * 100.0
        feox = float(row["fe_oxides"]) * 100.0

        txt = (f"Label {id2idx[sid]}\n"
               f"Carbonate : {carb:.0f}%\n"
               f"Silicate  : {sili:.0f}%\n"
               f"Clay      : {clay:.0f}%\n"
               f"Fe-oxides : {feox:.0f}%")

        # 오른쪽에 박스 위치를 균등 분배
        rank = pick_ids.index(sid) + 1
        box_x = W + 0.05 * W
        box_y = rank * (H / (len(pick_ids) + 1))

        ax1.annotate(
            txt,
            xy=(cx, cy), xycoords='data',
            xytext=(box_x, box_y), textcoords='data',
            fontsize=9, va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.0),
            arrowprops=dict(arrowstyle="-", lw=1.2, color="black"),
        )

    # 8) 범례(라벨 색상) — 너무 많으면 생략
    legend_ids = seg_ids[:max_legend]
    patches = [mpatches.Patch(color=np.array(color_for_idx(id2idx[s]))/255.0,
                              label=f"label {id2idx[s]}") for s in legend_ids]
    if patches:
        ax1.legend(handles=patches, loc="lower right", fontsize=8, frameon=True)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] visualization saved: {out_png}")



# =========================
# 5) 배치 실행 (코어ID별 집계) - 단일 정의
# =========================
# 배치 실행 함수: 코어ID별로 이미지 그룹핑 후 처리
def run_batch_grouped(
    image_dir,
    pseudo_label_dir,
    model_ckpt_path,
    output_dir,
    hrnet_variant='hrnet_w18',
    device='cuda',
    img_exts=('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'),
    verbose_mask=False
):
    os.makedirs(output_dir, exist_ok=True)

    # 쓰기 권한 확인
    try:
        _test_path = Path(output_dir) / "_write_test.tmp"
        with open(_test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(_test_path)
        print(f"[CHECK] write permission OK: {output_dir}")
    except Exception as e:
        print(f"[ERR] cannot write to output_dir: {output_dir}\n      -> {type(e).__name__}: {e}")
        return

    # 모델 로드
    print(f"[INFO] loading model: {model_ckpt_path}")
    model = HRNet_DeepLabV3(hrnet_variant=hrnet_variant, out_dim=3).to(device)
    try:
        try:
            state = torch.load(model_ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[ERR] model checkpoint load failed: {model_ckpt_path}\n      -> {type(e).__name__}: {e}")
        return
    print("[INFO] checkpoint loaded (strict=False).")

    # 이미지 재귀 수집
    image_paths = collect_images_recursive(image_dir, exts=img_exts)
    print(f"[INFO] images found: {len(image_paths)} in '{image_dir}'")
    if image_paths[:5]:
        print("[INFO] sample images:")
        for p in image_paths[:5]:
            print("       -", p)
    if not image_paths:
        print("[WARN] no images found. Stop.")
        return

    # 코어ID별 그룹핑
    imgs_by_core = {}
    for p in image_paths:
        fname = os.path.basename(p)
        cid = extract_core_id(fname)
        imgs_by_core.setdefault(cid, []).append(p)

    all_rows = []
    all_rows_nnls = []
    
    # 코어ID별 이미지 처리
    for core_id, paths in imgs_by_core.items():
        # 섹션 idx 정렬
        paths.sort(key=lambda full: (extract_section_idx(os.path.basename(full)), os.path.basename(full)))
        core_rows = []
        core_rows_nnls = []

        print(f"[INFO] processing core '{core_id}' with {len(paths)} images")

        for img_path in paths:
            fname = os.path.basename(img_path)
            base  = os.path.splitext(fname)[0]

            try:
                mask_path = find_mask_for_image(base, pseudo_label_dir, verbose=verbose_mask)
                print(f"[DBG] try mask for base='{base}' -> {mask_path}")
                if not mask_path:
                    print(f"[SKIP] mask not found for '{fname}' (search base='{base}') in '{pseudo_label_dir}'")
                    continue

                df_seg = segment_props_and_fractions(model, img_path, mask_path, device=device)
                if df_seg.empty:
                    print(f"[WARN] no usable segments for {fname} -> skip saving for this image")
                    continue

                # 메타 부여
                df_seg.insert(0, "core_id", core_id)
                df_seg.insert(1, "image_file", fname)
                df_seg.insert(2, "section_idx", extract_section_idx(fname))
                df_seg.insert(3, "base_name", base)
                df_seg.insert(4, "image_path", img_path)
                df_seg.insert(5, "mask_path", mask_path)

                # 이미지별 저장
                out_csv = Path(output_dir) / f"{base}_segments_fractions.csv"
                safe_to_csv(df_seg, out_csv, "per-image")

                # 이미지별 NNLS-only 저장
                out_csv_nnls = Path(output_dir) / f"{base}_nnls_only.csv"
                save_nnls_only(df_seg, out_csv_nnls, "per-image")

                # ▶ 이미지별 시각화 PNG 저장 (NEW)
                out_png = Path(output_dir) / f"{base}_viz.png"
                visualize_compositions(img_path, mask_path, df_seg, out_png, topk=3)

                core_rows.append(df_seg)

                nnls_cols = [c for c in NNLS_COLS if c in df_seg.columns]
                if nnls_cols:
                    core_rows_nnls.append(df_seg[nnls_cols].copy())

            except Exception as e:
                # 어떤 예외든 한 이미지에서만 스킵하고 계속 진행
                print(f"[ERR] unexpected error at image '{fname}'\n      -> {type(e).__name__}: {e}")
                continue

        # 코어 단위 집계
        if core_rows:
            core_df = pd.concat(core_rows, ignore_index=True)
            core_csv = Path(output_dir) / f"{core_id}_segments_fractions.csv"
            safe_to_csv(core_df, core_csv, "per-core")
            all_rows.append(core_df)

            if core_rows_nnls:
                core_df_nnls = pd.concat(core_rows_nnls, ignore_index=True)
                core_csv_nnls = Path(output_dir) / f"{core_id}_nnls_only.csv"
                safe_to_csv(core_df_nnls, core_csv_nnls, "per-core NNLS-only")
                all_rows_nnls.append(core_df_nnls)
        else:
            print(f"[INFO] no rows for core {core_id}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 전체 합본
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        all_csv = Path(output_dir) / "all_segments_fractions.csv"
        safe_to_csv(df_all, all_csv, "aggregate")

        if all_rows_nnls:
            df_all_nnls = pd.concat(all_rows_nnls, ignore_index=True)
            all_csv_nnls = Path(output_dir) / "all_nnls_only.csv"
            safe_to_csv(df_all_nnls, all_csv_nnls, "aggregate NNLS-only")
    else:
        print("[DONE] nothing processed. (No per-core data collected)")

# =========================
# 6) 실행 예시
# =========================
if __name__ == "__main__":
    image_dir        = r"C:\Users\Admin\Desktop\2nd core sample dataset\raw image"
    pseudo_label_dir = r"C:\Users\Admin\Desktop\2nd core sample dataset\slic_sam_labels"
    model_ckpt_path  = r"C:\Users\Admin\Desktop\Python\hrnet_dlv3_regression.pth"
    output_dir       = r"C:\Users\Admin\Desktop\2nd core sample dataset\nnls_out"

    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    run_batch_grouped(image_dir, pseudo_label_dir, model_ckpt_path, output_dir,
                      hrnet_variant='hrnet_w18', device=device, verbose_mask=False)
