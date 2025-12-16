import os, glob, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from controlnet_aux import LineartDetector

# =============================
# ====== 0. 기본 설정 ========
# =============================
SRC_DIR = Path("/home/aikusrv01/storyboard/dataset/Training/01.원천데이터/드라마")              # 원본 이미지 폴더
MID_DIR = Path("./Sample_merged_1024sq")       # 1024 정사각 변환 후 폴더
OUT_DIR = Path("./251112_SDXL_Lineart")        # SDXL-ControlNet 결과 저장 폴더

#JSON_LIST = Path("/home/aikusrv01/storyboard/dataset/Training/no_effect.jsonl")          # ← JSONL 입력 목록
JSON_LIST = Path("/home/aikusrv01/storyboard/TK/Dataset_close_shot/metadata.jsonl")          # ← JSONL 입력 목록


MID_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 1024
BG = (255, 255, 255)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =============================
# ===== JSONL 파일 읽기 =======
# =============================

def load_jsonl_filelist(json_path: Path):
    """
    JSON/JSONL 파일에서 file_name 필드를 읽어 파일명 목록을 리턴.
    확장자 포함 파일명 그대로 저장 + stem도 함께 저장하여 유연하게 매칭.
    """
    print(f"[INFO] Loading JSON list: {json_path}")

    full_names = set()   # ex) SF090738.JPEG
    stems      = set()   # ex) SF090738

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                continue

            if "file_name" not in obj:
                continue

            fname = obj["file_name"]

            # 확장자 포함된 형태
            full_names.add(fname)

            # 확장자 제거한 stem (JPG ↔ JPEG 경우도 대응)
            stems.add(Path(fname).stem)

    print(f"[INFO] Loaded {len(full_names)} image entries from JSON.")
    return full_names, stems

# =============================
# ====== 1. 정사각 1024 변환 =====
# =============================

def resize_to_1024(src: Path) -> Path:
    """이미지를 1024 정사각 + 여백으로 변환."""
    try:
        im = Image.open(src)
        im = ImageOps.exif_transpose(im).convert("RGB")

        w, h = im.size
        s = min(TARGET / w, TARGET / h)
        nw, nh = int(round(w * s)), int(round(h * s))

        im2 = im.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGB", (TARGET, TARGET), BG)
        off = ((TARGET - nw) // 2, (TARGET - nh) // 2)
        canvas.paste(im2, off)

        out = MID_DIR / f"{src.stem}.jpg"
        canvas.save(out, quality=92, subsampling=1, optimize=True)
        return out

    except Exception as e:
        print(f"[1024 FAIL] {src}: {e}")
        return None

def run_resize_batch(valid_files):
    print(f"[INFO] Filtering resize targets: {len(valid_files)} files")

    out_paths = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        for result in tqdm(ex.map(resize_to_1024, valid_files), total=len(valid_files)):
            if result is not None:
                out_paths.append(result)

    print(f"[INFO] Resized {len(out_paths)}/{len(valid_files)} images.")
    return out_paths

# =============================
# ===== 2. SDXL ControlNet =====
# =============================

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_ID = "ShermanG/ControlNet-Standard-Lineart-for-SDXL"
VAE_ID        = "madebyollin/sdxl-vae-fp16-fix"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT = "storyboard, clean pencil line sketch, cinematic framing, minimal shading, black and white, black sketch, white background"
NEG    = "color, heavy shading, photorealistic texture, noise, blur, artifacts"

STEPS    = 28
GUIDE    = 5.0
CN_SCALE = 0.9
SEED     = 1234


def load_pipeline():
    print("[INFO] Loading ControlNet…")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID, torch_dtype=DTYPE
    )

    print("[INFO] Loading SDXL pipeline…")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        vae=AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE),
        torch_dtype=DTYPE
    )

    pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()  # 메모리 최적화
    return pipe


# lineart extractor
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")

def normalize_edge(edge):
    """
    어떤 형식으로 edge가 와도 확실히 numpy uint8 3채널 이미지로 통일.
    SDXL ControlNet이 받아들일 수 있는 최종 형식으로 변환.
    """
    # 1. PIL.Image인 경우 → numpy로
    if isinstance(edge, Image.Image):
        edge = np.array(edge.convert("RGB"))

    # 2. bytes 로 올 경우 (일부 annotator는 PNG byte stream을 반환함)
    if isinstance(edge, bytes):
        import io
        edge = Image.open(io.BytesIO(edge)).convert("RGB")
        edge = np.array(edge)

    # 3. grayscale numpy (H,W) 형태면 채널 추가
    if isinstance(edge, np.ndarray):
        if edge.ndim == 2:
            edge = np.stack([edge] * 3, axis=-1)
        elif edge.ndim == 3 and edge.shape[2] == 1:
            edge = np.concatenate([edge]*3, axis=2)

        # 8비트 아닌 경우 맞춰주기
        if edge.dtype != np.uint8:
            edge = edge.astype(np.uint8)

        # numpy를 PIL로 변환
        return Image.fromarray(edge).convert("RGB")

    raise ValueError(f"Unsupported edge type: {type(edge)}")


def to_lineart_hint(pil_img: Image.Image, target_res=1024, coarse=False):
    w, h = pil_img.size
    if max(w, h) != target_res:
        scale = target_res / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(pil_img.convert("RGB"))
    edge = lineart(arr, coarse=coarse)
    hint = normalize_edge(edge)
    return hint


def run_controlnet(pipe, path: Path):
    try:
        name = path.stem
        img = Image.open(path).convert("RGB")  # 1024 정사각 전처리된 이미지

        # SDXL과 동일 해상도의 hint 생성
        hint = to_lineart_hint(img, target_res=TARGET, coarse=False)

        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

        out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG,
        image = img,
        controlnet_conditioning_image=hint,
        num_inference_steps=STEPS,
        guidance_scale=GUIDE,
        controlnet_conditioning_scale=CN_SCALE,
        generator=generator,
      )
        result = out.images[0]

        hint.save(OUT_DIR / f"{name}.jpeg")

        return True

    except Exception as e:
        print(f"[CN FAIL] {path}: {e}")
        return False


def run_controlnet_batch(img_list):
    pipe = load_pipeline()

    ok = 0
    for p in tqdm(img_list):
        ok += run_controlnet(pipe, p)

    print(f"[INFO] ControlNet processed {ok}/{len(img_list)} images.")


# 0. JSONL 파일에서 목록 불러오기
json_fullnames, json_stems = load_jsonl_filelist(JSON_LIST)

# 1. 디렉토리 전체 이미지 탐색
all_files = [p for p in SRC_DIR.rglob("*") if p.suffix.lower() in EXTS]

# 2. JSON에 있는 파일만 필터링 (확장자 다르더라도 매칭되게)
valid = []
for p in all_files:
    # 파일명 전체 일치
    if p.name in json_fullnames:
        valid.append(p)
        continue

    # 확장자 다르면 stem만 보고도 매칭 가능하게
    if p.stem in json_stems:
        valid.append(p)
        continue

print(f"[INFO] Found {len(valid)} valid images matching JSON list.")

# 3. 1024 정사각 변환
converted = run_resize_batch(valid)

# 4. ControlNet 처리
run_controlnet_batch(converted)