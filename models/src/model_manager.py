from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os


class ModelManager:
    """
    KoGPT2 모델, 토크나이저, 옵티마이저, 손실 함수를 통합 관리하는 클래스
    """
    def __init__(self, learning_rate=5e-5, epochs=3):
        """
        Args:
            learning_rate (float): 학습률
            epochs (int): 학습 Epoch 수
        """
        # ✅ 모델 및 토크나이저 초기화
        self.model_path = 'skt/kogpt2-base-v2'
        
        if not os.path.exists(self.model_path):
            print(f"⚠️ 경고: '{self.model_path}' 경로가 존재하지 않습니다. HuggingFace Hub에서 모델을 다운로드합니다.")
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.model_path,
            bos_token='<s>',
            eos_token='</s>',
            pad_token='<pad>',
            unk_token='<unk>',
            mask_token='<mask>'
        )
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        
        # ✅ 학습 설정
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = CrossEntropyLoss()
    
    def save(self, save_path: str):
        """
        모델과 토크나이저를 로컬 디렉토리에 저장합니다.
        
        Args:
            save_path (str): 저장할 디렉토리 경로
        """
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ 모델과 토크나이저가 '{save_path}'에 저장되었습니다.")
    
    def push_to_hub(self, repo_name: str):
        """
        모델과 토크나이저를 Hugging Face Hub에 업로드합니다.
        
        Args:
            repo_name (str): Hugging Face 리포지토리 이름
        """
        self.model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
        print(f"🚀 모델과 토크나이저가 Hugging Face Hub '{repo_name}'에 성공적으로 업로드되었습니다.")
