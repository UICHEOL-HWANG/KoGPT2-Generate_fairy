from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

class ModelManager:
    def __init__(self, learning_rate=5e-5, epochs=3):
        """
        KoGPT2 모델, 토크나이저, 옵티마이저, 손실 함수를 통합 관리하는 클래스
        
        Args:
            learning_rate (float): 학습률
            epochs (int): 학습 Epoch 수
        """
        # 모델 및 토크나이저 초기화
        
        self.model_path = 'skt/kogpt2-base-v2'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.model_path,
            bos_token='<s>',
            eos_token='</s>',
            pad_token='<pad>',
            unk_token='<unk>',
            mask_token='<mask>'
        )
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        
        # 학습 설정
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = CrossEntropyLoss()