import pandas as pd
from sklearn.model_selection import train_test_split
import json

def seperated_data(data):
    """
    데이터를 classification 기준으로 그룹화하여 일부를 샘플링한 후, 
    전체 데이터를 학습용(train)과 테스트용(test)으로 분할합니다.
    
    Args:
        data (list or dict): 원본 데이터
    
    Returns:
        pd.DataFrame: train 데이터프레임
        pd.DataFrame: test 데이터프레임
    """
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # 빈 데이터프레임 생성
    sep_data = pd.DataFrame()
    
    # 그룹별로 30% 샘플링
    for cls, group in df.groupby('classification'):
        sampled_group = group.sample(frac=0.3, random_state=42)
        sep_data = pd.concat([sep_data, sampled_group], ignore_index=True)
    
    # 학습용과 테스트용 데이터 분할
    train, test = train_test_split(sep_data, test_size=0.3, random_state=42)
    
    # 인덱스 리셋
    train = json.loads(train.reset_index(drop=True).to_json(orient="records"))
    test = json.loads(test.reset_index(drop=True).to_json(orient="records"))
    
    return train, test 
    
