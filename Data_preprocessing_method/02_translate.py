import json
import re
import torch
import sys
import os
from transformers import pipeline, AutoTokenizer

# =============================================================================
# [Configuration] 
# =============================================================================
CONFIG = {
    # 1. 처리 모드 선택 (주석을 해제/설정하여 모드를 변경하세요)
    # "ALL":     텍스트 전체를 한글로 간주하고 번역합니다.
    # "PARTIAL": 앞에 있는 영어(트리거 등)는 유지하고, 뒤에 나오는 한글만 번역합니다.
    "TRANSLATION_MODE": "PARTIAL", 
    # "TRANSLATION_MODE": "ALL",

    # 2. 처리할 파일 목록 (입력 파일 -> 출력 파일)
    "FILES_TO_PROCESS": [
        {
            "input": "/home/aikusrv01/storyboard/PSY/preprocessed_for_translation.jsonl",
            "output": "/home/aikusrv01/storyboard/PSY/final_translated_dataset.jsonl"
        }
    ],

    # 3. 모델 설정
    "MODEL_NAME": "gyupro/Koalpaca-Translation-KR2EN",
    "BATCH_SIZE": 4, # GPU 메모리에 따라 조절 (2~8)
}
# =============================================================================

def split_text_by_mode(text, mode):
    """
    설정된 모드에 따라 텍스트를 (보존할 앞부분, 번역할 뒷부분)으로 나눕니다.
    """
    # [모드 1] 부분 번역 (영어 유지, 한글만 번역)
    if mode == "PARTIAL":
        # 첫 번째 한글(가-힣)이 나오는 위치 탐색
        match = re.search(r'[가-힣]', text)
        if match:
            split_idx = match.start()
            
            # prefix: 영어/특수문자 (번역 X, 그대로 유지)
            prefix = text[:split_idx].strip()
            if prefix.endswith(','): prefix = prefix[:-1].strip()
            
            # suffix: 한글 포함 뒷부분 (번역 O)
            suffix = text[split_idx:].strip()
            
            return prefix, suffix
        else:
            # 한글이 없으면 전체를 prefix로 간주 (번역 안 함)
            return text, ""

    # [모드 2] 전체 번역 (텍스트 통째로 번역)
    elif mode == "ALL":
        # prefix 없음, 전체가 suffix(번역 대상)
        return "", text
    
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")

def build_prompt(korean_text):
    """Koalpaca 프롬프트 생성"""
    return f"Korean: {korean_text}\nEnglish:"

def main():
    print(f"🚀 번역 파이프라인 시작 (모드: {CONFIG['TRANSLATION_MODE']})")
    
    # 1. 모델 로드 (최초 1회)
    print(f"   - 모델 로딩 중: {CONFIG['MODEL_NAME']}")
    try:
        device_index = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        
        # 8-bit 로딩이 필요한 경우 load_in_8bit=True 사용 (bitsandbytes 필요)
        generator = pipeline(
            "text-generation",
            model=CONFIG['MODEL_NAME'],
            tokenizer=tokenizer,
            device=device_index,
            torch_dtype=torch.float16
        )
        print("   ✅ 모델 로드 완료.\n")
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
        sys.exit(1)

    # 2. 파일 목록 순회
    for idx, file_info in enumerate(CONFIG['FILES_TO_PROCESS']):
        input_path = file_info['input']
        output_path = file_info['output']
        
        print(f"📄 [파일 {idx+1}/{len(CONFIG['FILES_TO_PROCESS'])}] 처리 시작")
        print(f"   - 입력: {os.path.basename(input_path)}")

        all_items = []
        texts_to_translate = [] # 실제로 모델에 들어갈 텍스트 리스트
        item_indices = []       # 나중에 결과를 매핑하기 위한 인덱스

        # 2-1. 파일 읽기 및 데이터 분리
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    original_text = data.get('text', '')

                    # 모드에 따라 분리 (보존할 앞부분 / 번역할 뒷부분)
                    prefix, suffix = split_text_by_mode(original_text, CONFIG['TRANSLATION_MODE'])
                    
                    item = {
                        'file_name': data['file_name'],
                        'prefix': prefix,
                        'suffix': suffix,
                        'translated_suffix': ''
                    }
                    all_items.append(item)

                    # 번역할 내용이 있는 경우에만 리스트에 추가
                    if suffix:
                        texts_to_translate.append(suffix)
                        item_indices.append(len(all_items) - 1)
            
            print(f"   - 데이터 로드: 총 {len(all_items)}개 (번역 대상: {len(texts_to_translate)}개)")

        except FileNotFoundError:
            print(f"   🚨 파일을 찾을 수 없어 건너뜁니다: {input_path}")
            continue

        # 2-2. 배치 번역 실행
        if texts_to_translate:
            print(f"   - 번역 수행 중 (Batch Size: {CONFIG['BATCH_SIZE']})...")
            
            prompts = [build_prompt(t) for t in texts_to_translate]
            total_prompts = len(prompts)
            
            for i in range(0, total_prompts, CONFIG['BATCH_SIZE']):
                batch_prompts = prompts[i : i + CONFIG['BATCH_SIZE']]
                
                try:
                    outputs = generator(
                        batch_prompts,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        return_full_text=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    for j, output in enumerate(outputs):
                        generated_text = output[0]['generated_text']
                        
                        # 결과 파싱 ("English:" 뒷부분 추출)
                        if "English:" in generated_text:
                            translation = generated_text.split("English:", 1)[1].strip()
                        else:
                            translation = generated_text.strip()
                        
                        # 원본 아이템에 결과 매핑
                        global_idx = item_indices[i + j]
                        all_items[global_idx]['translated_suffix'] = translation

                    # 진행 상황 출력 (간략하게)
                    current = min(i + CONFIG['BATCH_SIZE'], total_prompts)
                    if (current % 100 == 0) or (current == total_prompts):
                        print(f"     ... {current}/{total_prompts} 완료")

                except Exception as e:
                    print(f"     🚨 배치 번역 오류 ({i}~): {e}")

        # 2-3. 결과 병합 및 저장
        print(f"   - 저장 중: {os.path.basename(output_path)}")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for item in all_items:
                    prefix = item['prefix']
                    trans = item['translated_suffix']
                    
                    # 최종 텍스트 조립
                    if prefix and trans:
                        final_text = f"{prefix}, {trans}"
                    elif prefix:
                        final_text = prefix
                    elif trans:
                        final_text = trans
                    else:
                        final_text = ""
                    
                    new_entry = {
                        "file_name": item['file_name'],
                        "text": final_text
                    }
                    json.dump(new_entry, f_out, ensure_ascii=False)
                    f_out.write('\n')
            print("   ✅ 저장 완료!\n")
            
        except Exception as e:
            print(f"   🚨 파일 저장 실패: {e}")

    print("🎉 모든 번역 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()