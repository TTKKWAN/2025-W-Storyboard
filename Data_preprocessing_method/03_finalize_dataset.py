import json
import os
import sys
from transformers import AutoTokenizer

# =============================================================================
# [Configuration] 
# =============================================================================
CONFIG = {
    # 1. 처리할 파일 목록 (입력 -> 출력)
    "FILES_TO_PROCESS": [
        {
            "input": "/home/aikusrv01/storyboard/PSY/final_translated_dataset.jsonl",
            "output": "/home/aikusrv01/storyboard/PSY/train_dataset_final.jsonl"
        }
    ],

    # 2. 토큰 필터링 설정
    "TOKEN_LIMIT": 77,     # CLIP 모델 기준 (일반적으로 75~77)
    "TOKENIZER_MODEL": "openai/clip-vit-large-patch14", # 토크나이저 로드용 모델명

    # 3. 태그 재정렬 및 트리거 설정
    "TARGET_SHOTS": ['medium shot', 'close-up shot', 'full shot'],
    "TARGET_ANGLES": ['High angle', 'Low angle', 'Eye level'],
    "TRIGGER_MAP": {
        'medium shot': '<ms_trg>',
        'close-up shot': '<cu_trg>',
        'full shot': '<fs_trg>'
    }
}
# =============================================================================

def reorder_and_add_trigger(text):
    """
    [기능 1] 텍스트 포맷팅
    - 샷/앵글 태그를 찾아 순서를 [트리거, 샷, 앵글, ...]로 변경합니다.
    - 트리거 워드를 맨 앞에 부착합니다.
    """
    tags = [t.strip() for t in text.split(',')]
    
    found_shot = None
    found_angle = None
    other_tags = []
    
    # 태그 분류
    for tag in tags:
        if tag in CONFIG['TARGET_SHOTS']:
            found_shot = tag
        elif tag in CONFIG['TARGET_ANGLES']:
            found_angle = tag
        else:
            # 이미 트리거 형식이면 중복 방지 위해 제외, 아니면 기타 태그로 분류
            if not (tag.startswith('<') and '_trg>' in tag):
                other_tags.append(tag)
    
    # 샷 정보가 없으면 원본 그대로 반환
    if not found_shot:
        return text, False

    # 트리거 결정
    trigger = CONFIG['TRIGGER_MAP'].get(found_shot, '')
    
    # 순서 재조립: [트리거] -> [샷] -> [앵글] -> [나머지]
    new_tags = [trigger, found_shot]
    if found_angle:
        new_tags.append(found_angle)
    new_tags.extend(other_tags)
    
    return ", ".join(new_tags), True

def main():
    print("🚀 최종 데이터셋 확정 파이프라인을 시작합니다.")
    print(f"   - 토큰 제한: {CONFIG['TOKEN_LIMIT']} (Model: {CONFIG['TOKENIZER_MODEL']})")

    # 1. 토크나이저 로드 (한 번만 수행)
    print("   - 토크나이저 로딩 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['TOKENIZER_MODEL'])
    except Exception as e:
        print(f"🚨 토크나이저 로드 실패: {e}")
        sys.exit(1)

    # 2. 파일 처리 루프
    for file_info in CONFIG['FILES_TO_PROCESS']:
        input_path = file_info['input']
        output_path = file_info['output']
        
        print(f"\n📄 처리 중: {os.path.basename(input_path)}")
        
        processed_items = []
        stats = {
            "total_read": 0,
            "triggered": 0,
            "filtered_length": 0,
            "final_saved": 0
        }

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    stats['total_read'] += 1
                    
                    item = json.loads(line)
                    original_text = item.get('text', '')

                    # [단계 1] 포맷팅 (트리거 추가 및 정렬)
                    formatted_text, is_triggered = reorder_and_add_trigger(original_text)
                    if is_triggered:
                        stats['triggered'] += 1
                    
                    # [단계 2] 필터링 (토큰 길이 검사)
                    # 주의: 포맷팅된 텍스트(formatted_text) 기준으로 길이를 재야 함!
                    token_ids = tokenizer(formatted_text, add_special_tokens=True).input_ids
                    if len(token_ids) > CONFIG['TOKEN_LIMIT']:
                        stats['filtered_length'] += 1
                        continue # 저장하지 않고 건너뜀

                    # [단계 3] 저장 목록에 추가
                    item['text'] = formatted_text
                    processed_items.append(item)
                    stats['final_saved'] += 1

            # 파일 저장
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for item in processed_items:
                    json.dump(item, f_out, ensure_ascii=False)
                    f_out.write('\n')
            
            # 결과 리포트
            print(f"📊 결과 요약:")
            print(f"   - 읽은 데이터: {stats['total_read']}개")
            print(f"   - 트리거 부착: {stats['triggered']}개")
            print(f"   ❌ 길이 초과 제외: {stats['filtered_length']}개")
            print(f"   ✅ 최종 저장: {stats['final_saved']}개")
            print(f"   📂 저장 경로: {output_path}")

        except FileNotFoundError:
            print(f"🚨 파일을 찾을 수 없습니다: {input_path}")
            continue

    print("\n🎉 모든 작업 완료! 학습 가능한 최종 데이터셋이 준비되었습니다.")

if __name__ == "__main__":
    main()