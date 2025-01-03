import json
import re

# 📂 입력 및 출력 파일 설정
base_root = './valid_dataset.json'
output_root = './cleaned_valid_dataset.json'

# 📝 텍스트 클리닝 함수
def cleaned_text(text):
    return re.sub(r'\\', '', text)

# 📦 JSON 데이터 클리닝
def clean_json_data(data):
    for lines in data:
        if 'mergedText' in lines:
            lines['mergedText'] = cleaned_text(lines['mergedText'])
    return data  # 수정된 데이터를 반환

# 🚀 메인 실행
if __name__ == '__main__':
    try:
        # JSON 데이터 읽기
        with open(base_root, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # JSON 데이터 클리닝
        cleaned_data = clean_json_data(data)
        
        # 결과 저장
        with open(output_root, 'w', encoding='utf-8') as file:
            json.dump(cleaned_data, file, ensure_ascii=False, indent=2)
        
        print(f"[✅] 클리닝 완료: {output_root}")
    
    except FileNotFoundError:
        print(f"[❌] 파일을 찾을 수 없습니다: {base_root}")
    except KeyError as e:
        print(f"[❌] 키 오류: {e}")
    except Exception as e:
        print(f"[❌] 기타 오류: {e}")
