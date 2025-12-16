# Storyboard Trigger Project

## 1. 프로젝트 소개
`github_ver`는 촬영 콘티 이미지를 기반으로 스토리보드 스타일의 장면을 합성하기 위한 엔드 투 엔드 파이프라인입니다. 원시 데이터 수집 → 전처리/라인아트 생성 → 샷 클러스터링 → LoRA + 신규 토큰 학습 → 추론 및 검증까지의 흐름을 한 곳에서 재현할 수 있도록 정리했습니다.

## 2. 디렉터리 구성
| 경로 | 설명 |
| --- | --- |
| `Dataset_collect_title_method/` | 원본 스틸컷 메타데이터 정리 및 캡션/타이틀 매칭 스크립트 |
| `Data_preprocessing_method/`, `Data_preprocessing_Line/` | 말풍선 제거, 1024 정사각 변환, SDXL ControlNet 라인아트 생성 등 데이터 정제 단계 |
| `Dataset_clustering_method/` | CLIP 임베딩 기반 장면/샷 타입 클러스터링 코드 및 사전 계산된 가중치 |
| `Dataset_fin/` | (GitHub에는 제외) 모델 학습에 사용한 최종 이미지/JSONL. 실제 운용 시 로컬 동일 경로에 배치 |
| `train_trigger02.py`, `shot_trigger_train.sh` | LoRA 및 커스텀 Trigger Token 학습 스크립트와 실행 셸 |
| `inference_trigger.py`, `inference_trigger2.sh` | 학습된 토큰과 LoRA를 사용해 이미지를 합성하는 추론 파이프라인 |
| `validation/` | 검증용 JSONL, 배치 생성 스크립트, 멀티 GPU 실행 셸 포함 |
| `model_sd/`, `model_output/` | (GitHub에는 제외) 베이스 Stable Diffusion 체크포인트와 생성 결과 저장소 |

> ⚠️ `model_sd/`, `model_output/`, `Dataset_fin/` 등 대용량 파일은 `.gitignore`에 등록되어 있으므로 공개 저장소에는 커밋되지 않습니다. 필요한 경우 동일한 디렉터리명으로 로컬에 배치한 뒤 스크립트를 실행하세요.

## 3. 방법론 요약
### 3.1 데이터 수집 및 정규화
1. `Dataset_collect_title_method/collect_title_matches.py`로 원본 JSONL을 통합하고, 중복 캡션/누락 항목을 정리합니다.
2. `Data_preprocessing_method/` 스크립트와 `Data_preprocessing_Line/01_preprocessing.py`를 사용해 1024x1024 정사각 이미지와 라인아트 버전을 생성합니다. 이때 ControlNet(Lineart) + SDXL 파이프라인을 활용해 노이즈를 최소화합니다.
3. 필요 시 `00_Canny.ipynb`로 거친 에지/버블 제거 결과를 확인합니다.

### 3.2 샷 분류 및 데이터 필터링
- `Dataset_clustering_method/run_clip_clustering.py` + 사전 학습된 CLIP 가중치(`ViT-B-32-openai.pt`, `clip-vit-b-32.pt`)를 이용해 장면 별 샷 타입을 자동 분류하고, 균형 잡힌 학습 세트를 만듭니다.
- 결과는 `Dataset_fin/metadata.jsonl`로 요약되며, 학습 스크립트에서 그대로 사용합니다.

### 3.3 Trigger Token + LoRA 학습
- `train_trigger02.py`는 diffusers 기반 LoRA 학습 스크립트이며 `<fs_trg>`, `<ms_trg>` 등 커스텀 토큰을 토크나이저에 추가하고 텍스트 인코더를 동시에 미세조정합니다.
- `shot_trigger_train.sh`에서 주요 하이퍼파라미터(데이터 경로, 배치, rank 등)를 지정하고 `accelerate` 실행 구성을 로깅합니다.

### 3.4 추론 및 검증
- `inference_trigger.py`는 학습된 LoRA + 텍스트 인코더 가중치를 로드해 사용자 프롬프트에 트리거 단어를 삽입한 뒤 결과 이미지를 저장합니다.
- `validation/validation.sh`는 여러 GPU에 샤딩하여 검증용 JSONL(`validation.jsonl`)에 있는 프롬프트를 일괄 생성합니다. 필요 시 `validation/base_valid.sh`로 LoRA 없이 베이스라인을 비교합니다.

## 4. 환경 설정
1. Python 3.10+ 환경을 권장합니다.
2. 가상환경 생성 및 의존성 설치:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows라면 .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. NVIDIA GPU + CUDA 11.8 이상, VRAM 24GB 이상의 환경을 권장합니다. `xformers` 설치는 GPU/드라이버에 따라 추가 설정이 필요할 수 있습니다.

## 5. 실행 가이드
### 5.1 데이터 전처리
```bash
cd Data_preprocessing_method
python 01_preprocess_tags.py --input metadata_raw.jsonl --output metadata_clean.jsonl
python 02_translate.py --input metadata_clean.jsonl --output metadata_translated.jsonl
python 03_finalize_dataset.py --image-root /path/to/images --metadata metadata_translated.jsonl --out ../Dataset_fin
```
필요한 ControlNet 라인아트 생성은 `Data_preprocessing_Line/01_preprocessing.py` 내 상단 경로를 실제 데이터 경로로 수정 후 실행합니다.

### 5.2 트리거 LoRA 학습
```bash
bash shot_trigger_train.sh \
  --pretrained_model_name_or_path ./model_sd \
  --train_data_dir ./Dataset_fin \
  --output_dir ./model_sd_weight/train_fs
```
`shot_trigger_train.sh` 내부에서 `accelerate launch train_trigger02.py`가 호출되며, `NEW_TOKENS`, `TOKEN_INITIALIZER` 설정을 그대로 사용합니다.

### 5.3 검증 및 추론
- 검증: `validation/validation.sh`의 경로(GPU, 모델, 체크포인트)를 로컬 환경에 맞게 수정한 뒤 실행합니다.
  ```bash
  cd validation
  bash validation.sh
  ```
- 단일 프롬프트 추론:
  ```bash
  python inference_trigger.py \
    --base-model ./model_sd \
    --checkpoint ./model_sd_weight/train_fs \
    --prompt "<fs_trg>, full shot, Eye level, outdoor" \
    --output ./model_output/sample.png
  ```

## 6. 추가 팁
- 대용량 데이터나 가중치는 Git LFS 대신 로컬/사내 스토리지에 두고, README에 다운로드 경로만 안내하는 것을 권장합니다.
- `wandb`를 사용할 경우 `WANDB_API_KEY` 환경 변수를 설정하고 `train_trigger02.py --report_to wandb` 옵션을 켜면 실험 기록을 자동화할 수 있습니다.
- `validation` 결과는 `validation/output/` 아래에 체크포인트별로 저장되며 `.gitignore`에 의해 커밋되지 않습니다.
