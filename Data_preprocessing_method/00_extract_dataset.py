import json
import os
import glob
from collections import defaultdict

# =============================================================================
# [Configuration] 
# =============================================================================
CONFIG = {
    # 1. 라벨링 데이터가 위치한 최상위 경로 (dataset/Validation/02.라벨링데이터 등)
    "BASE_LABEL_DIR": "/home/aikusrv01/storyboard/dataset/Validation/02.라벨링데이터", 
    
    # 2. 처리할 대상 (특정 폴더만 하려면 리스트에 추가, 전체를 하려면 빈 리스트 [])
    "TARGET_FOLDERS": [
        'VL_01._로맨스',
        'VL_02._드라마',
        'VL_09._감성'   
    ], 

    # 3. 결과 파일 저장 경로
    "OUTPUT_FILE": "/home/aikusrv01/storyboard/PSY/validation.jsonl",

    # 4. 필터링 조건
    "MAX_CHAR_NUM": 1,      # 최대 인원 수 (이 숫자보다 많으면 제외)
}
# =============================================================================

def extract_tags(label_data):
    """
    JSON 데이터에서 학습에 필요한 핵심 태그만 순서대로 추출합니다.
    Composition -> Character(Gender, Age, Emotion, Clothing, Props, Movement) -> Background_info
    """
    ordered_tags = []

    # 1. Composition (구도/앵글/조명)
    comp = label_data.get('directing', {}).get('composition', {})
    if comp:
        ordered_tags.extend(comp.values())

    # 2. Character Info (성별, 나이, 감정, 의상, 소품, 동작)
    char_list = label_data.get('character', {}).get('char_info', [])
    for char in char_list:
        # 기본 정보
        if char.get('gender'): ordered_tags.append(char.get('gender'))
        if char.get('age'): ordered_tags.append(char.get('age'))

        # 감정 (shape의 3번째 요소)
        shape_str = char.get('shape', "")
        if shape_str:
            parts = shape_str.split(',')
            if len(parts) > 2 and parts[2].strip():
                ordered_tags.append(parts[2].strip())

        # 의상, 소품, 동작 (쉼표로 구분된 데이터 처리)
        for key in ['clothing', 'props', 'movement']:
            val = char.get(key, "")
            if val:
                ordered_tags.extend([t.strip() for t in val.split(',') if t.strip()])

    # 3. Background (첫 번째 '배경 유무' 태그 제외하고 장소 정보만)
    bg_info = label_data.get('background', {}).get('background_info', "")
    if bg_info:
        bg_parts = bg_info.split(',')
        if len(bg_parts) > 1:
            ordered_tags.extend([t.strip() for t in bg_parts[1:] if t.strip()])

    # 빈 태그 제거 및 쉼표로 합쳐서 반환
    return ", ".join([t for t in ordered_tags if t and t.strip()])

def main():
    print(f"🚀 데이터 추출을 시작합니다.")
    print(f"   - 대상 경로: {CONFIG['BASE_LABEL_DIR']}")
    print(f"   - 인원 제한: {CONFIG['MAX_CHAR_NUM']}명 이하")

    # 1. 파일 리스트 확보
    search_patterns = []
    if CONFIG['TARGET_FOLDERS']:
        # 특정 폴더만 지정한 경우
        for folder in CONFIG['TARGET_FOLDERS']:
            path = os.path.join(CONFIG['BASE_LABEL_DIR'], folder, '**', '*.json')
            search_patterns.append(path)
    else:
        # 전체 폴더 스캔 (기본)
        path = os.path.join(CONFIG['BASE_LABEL_DIR'], '**', '*.json')
        search_patterns.append(path)

    json_files = []
    for pattern in search_patterns:
        json_files.extend(glob.glob(pattern, recursive=True))

    print(f"   - 발견된 파일 수: {len(json_files)}개")

    # 2. 데이터 처리
    processed_data = []
    stats = defaultdict(int)

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            label_data = content.get('label', {})

            # [필터] 인원 수 체크 (핵심 로직)
            char_num = label_data.get('character', {}).get('char_num', 0)
            if char_num > CONFIG['MAX_CHAR_NUM']:
                stats['skipped_char_limit'] += 1
                continue

            # [추출] 태그 정보 추출
            extracted_text = extract_tags(label_data)
            
            if not extracted_text:
                stats['skipped_no_tags'] += 1
                continue

            # [변환] 파일명 변환 (L...json -> S...JPEG)
            json_name = os.path.basename(file_path)
            # 파일명이 L로 시작하는 경우에만 변환 수행
            if json_name.startswith('L'):
                image_name = json_name.replace('L', 'S', 1).replace('.json', '.JPEG')
                
                processed_data.append({
                    "file_name": image_name,
                    "text": extracted_text
                })

        except Exception:
            stats['errors'] += 1

    # 3. 결과 저장
    # 저장 경로의 폴더가 없으면 생성
    os.makedirs(os.path.dirname(CONFIG['OUTPUT_FILE']), exist_ok=True)
    
    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        for item in processed_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    # 4. 결과 리포트
    print("\n" + "="*40)
    print("📊 [처리 결과]")
    print(f"   ✅ 저장 완료: {len(processed_data)}개")
    print(f"   ❌ 인원 초과로 제외: {stats['skipped_char_limit']}개")
    print(f"   📂 저장된 파일: {CONFIG['OUTPUT_FILE']}")
    print("="*40)

if __name__ == "__main__":
    main()