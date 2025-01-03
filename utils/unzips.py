import os
import zipfile

base_dir = './train/02.라벨링데이터/'


# 모든 ZIP 파일 해제
def unzip_all_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                extract_to_path = os.path.join(root, os.path.splitext(file)[0])  # 압축 해제 폴더 생성
                
                # 폴더가 존재하지 않으면 생성
                os.makedirs(extract_to_path, exist_ok=True)
                
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to_path)
                        print(f"[✅] 해제 성공: {zip_file_path} → {extract_to_path}")
                except zipfile.BadZipFile:
                    print(f"[❌] 오류: {zip_file_path} - 올바르지 않은 ZIP 파일입니다.")
                except Exception as e:
                    print(f"[❌] 오류: {zip_file_path} - {e}")
                    
                    

if __name__ == '__main__':
    unzip_all_files(base_dir=base_dir)