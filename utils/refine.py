import json
import os 

base_dir = './valid/02.라벨링데이터/'
OUTPUT_FILE = "valid_dataset.json"  # 병합된 JSON 파일명


def extract_metadata_and_merge_text(data):
    metadata = {
        "title": data.get("title"),
        "classification": data['paragraphInfo'][0]['plotSummaryInfo']['classification'],
        "readAge": data['paragraphInfo'][0]['plotSummaryInfo']['readAge'],
        "mergedText": " ".join(
            paragraph["srcText"]
            for paragraph in sorted(data.get("paragraphInfo", []), key=lambda x: x["srcPage"])
        )
    }
    return metadata

# 📦 모든 JSON 파일 읽기 및 병합
def process_json_files_to_list(base_dir, output_file):
    merged_data = []  # 모든 데이터를 저장할 리스트
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                
                try:
                    # JSON 파일 읽기
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 메타데이터와 병합된 텍스트 생성
                    processed_data = extract_metadata_and_merge_text(data)
                    merged_data.append(processed_data)
                    
                    print(f"[✅] 성공: {json_file_path}")
                
                except KeyError as e:
                    print(f"[❌] 오류: {json_file_path} - 필수 키가 존재하지 않습니다: {e}")
                except Exception as e:
                    print(f"[❌] 오류: {json_file_path} - {e}")
    
    # 하나의 JSON 파일로 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(merged_data, out_f, ensure_ascii=False, indent=2)
        print(f"[🚀] 모든 데이터가 성공적으로 {output_file}에 저장되었습니다!")
    except Exception as e:
        print(f"[❌] 오류: 최종 병합 JSON 저장 중 오류 발생: {e}")
                    

if __name__ == '__main__':
    process_json_files_to_list(base_dir, OUTPUT_FILE)