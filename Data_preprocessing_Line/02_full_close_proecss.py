import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from controlnet_aux import LineartDetector

# =============================
# ====== 0. 기본 설정 ========
# =============================

SRC_DIR = Path("/home/aikusrv01/storyboard/SY/collect_title")
OUT_DIR = Path("/home/aikusrv01/storyboard/SY/251112_SDXL_Lineart")
JSON_PATH = Path("/home/aikusrv01/storyboard/SY/0to2_eng.jsonl")   # ★ JSONL 경로 추가

OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================
# ====== 1. JSON 읽기 =========
# =============================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


json_data = load_jsonl(JSON_PATH)

# file_name → text 매핑
json_map = {item["file_name"]: item["text"].lower() for item in json_data}

# =============================
# ====== 2. 모델 로드 =========
# =============================

print("[INFO] Loading Standard Lineart Detector (Coarse Mode)...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
processor.to(DEVICE)

# =============================
# ====== 3. 처리 함수 =========
# =============================

def should_process(img_name: str) -> bool:
    """
    JSON 내 text에서 full shot 또는 close-up shot 포함 여부 판단
    """
    if img_name not in json_map:
        return False

    text = json_map[img_name]
    if  "close-up shot" in text:
        return True
    return False


def process_lineart(img_path: Path):
    try:
        image = Image.open(img_path).convert("RGB")

        detect_res = 512

        detected_map = processor(
            image,
            detect_resolution=detect_res,
            image_resolution=image.size[0],
        )

        save_path = OUT_DIR / img_path.name
        detected_map.save(save_path)

        return True

    except Exception as e:
        print(f"[FAIL] {img_path.name}: {e}")
        return False


# =============================
# ====== 4. 메인 실행 ========
# =============================

def main():
    all_files = [p for p in SRC_DIR.glob("*") if p.suffix.lower() in EXTS]

    print(f"[INFO] 총 입력 이미지: {len(all_files)}장")

    # JSON 기반 필터링
    target_files = [p for p in all_files if should_process(p.name)]

    print(f"[INFO] 'full shot' 또는 'close-up shot' 포함된 이미지: {len(target_files)}장만 처리")

    success_count = 0
    for img_path in tqdm(target_files):
        if process_lineart(img_path):
            success_count += 1

    print("-" * 30)
    print(f"[완료] 총 {success_count}/{len(target_files)}장 처리 완료.")

if __name__ == "__main__":
    main()
