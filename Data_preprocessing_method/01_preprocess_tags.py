import json
import os
from collections import Counter

# =============================================================================
# [Configuration] 
# =============================================================================
CONFIG = {
    # 1. 입력 파일 (데이터 추출 단계에서 생성된 파일)
    "INPUT_FILE": "/home/aikusrv01/storyboard/PSY/validation.jsonl",
    
    # 2. 출력 파일 (전처리 완료 후 번역 단계로 넘어갈 파일)
    "OUTPUT_FILE": "/home/aikusrv01/storyboard/PSY/validation_11.jsonl",

    # 3. 앵글(Angle) 변환 규칙: {한글: 영어}
    "ANGLE_MAP": {
        '하이앵글': 'High angle',
        '로우앵글': 'Low angle',
        '아이레벨': 'Eye level'
    },

    # 4. 샷(Shot) 변환 규칙: {한글: 영어}
    # (롱 샷과 풀 샷을 'full shot'으로 통합하는 로직 포함)
    "SHOT_MAP": {
        '클로즈업 샷': 'close-up shot',
        '미디엄 샷': 'medium shot',
        '롱 샷': 'full shot',
        '풀 샷': 'full shot'
    },

    # 5. 제거할 태그 접미사 (예: '...조명'으로 끝나는 태그 삭제)
    "REMOVE_SUFFIX": "조명"
}
# =============================================================================

def preprocess_text(text):
    """
    텍스트(태그 문자열)를 입력받아 다음 작업을 수행합니다:
    1. 불필요한 태그(조명 등) 제거
    2. 앵글/샷 용어를 영어 표준 용어로 변환
    3. 나머지 태그는 그대로 유지
    """
    if not text:
        return ""

    tags = [t.strip() for t in text.split(',')]
    new_tags = []
    
    for tag in tags:
        # [규칙 1] 특정 접미사(조명)가 있는 태그 제거
        if CONFIG["REMOVE_SUFFIX"] and tag.endswith(CONFIG["REMOVE_SUFFIX"]):
            continue
        
        # [규칙 2] 앵글 변환
        if tag in CONFIG["ANGLE_MAP"]:
            new_tags.append(CONFIG["ANGLE_MAP"][tag])
            continue
        
        # [규칙 3] 샷 변환
        if tag in CONFIG["SHOT_MAP"]:
            new_tags.append(CONFIG["SHOT_MAP"][tag])
            continue
        
        # [규칙 4] 그 외 태그는 유지
        new_tags.append(tag)
    
    return ", ".join(new_tags)

def main():
    print("🚀 태그 전처리 및 표준화 작업을 시작합니다.")
    print(f"   - 입력: {CONFIG['INPUT_FILE']}")
    print(f"   - 조명 정보('{CONFIG['REMOVE_SUFFIX']}') 제거: 활성화됨")
    
    processed_data = []
    stats = {
        "angle_count": Counter(),
        "shot_count": Counter(),
        "total_lines": 0
    }

    try:
        with open(CONFIG['INPUT_FILE'], 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                
                item = json.loads(line)
                original_text = item.get('text', '')
                
                # 전처리 수행
                new_text = preprocess_text(original_text)
                
                # 통계 집계 (변환된 태그 기준)
                for tag in new_text.split(', '):
                    if tag in CONFIG["ANGLE_MAP"].values():
                        stats["angle_count"][tag] += 1
                    elif tag in CONFIG["SHOT_MAP"].values():
                        stats["shot_count"][tag] += 1
                
                processed_data.append({
                    "file_name": item['file_name'],
                    "text": new_text
                })
                stats["total_lines"] += 1

    except FileNotFoundError:
        print(f"🚨 오류: 입력 파일 '{CONFIG['INPUT_FILE']}'을 찾을 수 없습니다.")
        exit()

    # 결과 저장
    os.makedirs(os.path.dirname(CONFIG['OUTPUT_FILE']), exist_ok=True)
    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        for item in processed_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    # 리포트 출력
    print("\n" + "="*40)
    print("📊 [전처리 결과 리포트]")
    print(f"   ✅ 처리된 데이터: {stats['total_lines']}개")
    print(f"   📂 저장 경로: {CONFIG['OUTPUT_FILE']}")
    
    print(f"\n🎥 앵글(Angle) 분포:")
    for angle, count in stats["angle_count"].most_common():
        print(f"   - {angle}: {count}개")
        
    print(f"\n🎬 샷(Shot) 분포:")
    for shot, count in stats["shot_count"].most_common():
        print(f"   - {shot}: {count}개")
    print("="*40)

if __name__ == "__main__":
    main()