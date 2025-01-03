from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset
import re 

def split_text_into_chunks_with_metadata(dataset, tokenizer: PreTrainedTokenizerFast, max_length=500, overlap=1):
    """
    긴 텍스트를 문장 단위로 자연스럽게 분할하면서 각 청크에 <s>, </s> 토큰을 추가하는 함수

    Args:
        dataset (list): 원본 JSON 데이터셋 (title, classification, readAge, mergedText 포함)
        tokenizer (PreTrainedTokenizerFast): KoGPT2 토크나이저 객체
        max_length (int): 청크당 최대 토큰 길이
        overlap (int): 청크 간 겹치는 문장 수

    Returns:
        list: <s>, </s> 토큰이 추가된 텍스트 청크 리스트
    """
    chunks = []

    for data in dataset:
        # 메타데이터 생성
        metadata = f"title: {data['title']}, classification: {data['classification']}, readAge: {data['readAge']}"
        text = f"{metadata}\n{data.get('mergedText', '')}"

        # 문장 단위로 분할
        sentences = re.split(r'(?<=[.!?])\s+', text)  # 문장 종결 기호로 분할
        current_chunk = []
        chunk_id = 1

        for sentence in sentences:
            temp_chunk = " ".join(current_chunk + [sentence])
            temp_chunk_with_tokens = f"<s> {temp_chunk.strip()} </s>"
            tokenized_length = len(tokenizer.encode(temp_chunk_with_tokens, add_special_tokens=False))

            if tokenized_length <= max_length:
                current_chunk.append(sentence)
            else:
                # 현재 청크 저장 (BOS/EOS 추가)
                final_chunk = f"<s> {' '.join(current_chunk).strip()} </s>"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": final_chunk
                })
                chunk_id += 1
                # 새로운 청크 시작 (Overlap 적용)
                current_chunk = current_chunk[-overlap:] + [sentence]

        # 마지막 청크 추가 (BOS/EOS 추가)
        if current_chunk:
            final_chunk = f"<s> {' '.join(current_chunk).strip()} </s>"
            chunks.append({
                "chunk_id": chunk_id,
                "text": final_chunk
            })

    return chunks

class KoGPTDataset(Dataset):
    """
    KoGPT2 학습을 위한 PyTorch Dataset 클래스
    """
    def __init__(self, chunks, tokenizer: PreTrainedTokenizerFast, max_length=500, return_chunk_id=False):
        """
        Args:
            chunks (list): 분할된 텍스트 청크 리스트
            tokenizer (PreTrainedTokenizerFast): 토크나이저 객체
            max_length (int): 최대 토큰 길이
            return_chunk_id (bool): 청크 ID 반환 여부 (디버깅용)
        """
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_chunk_id = return_chunk_id
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        text = chunk['text']

        encoded = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        sample = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0)  # input_ids를 labels로 복제
        }
        
        if self.return_chunk_id:
            sample['chunk_id'] = chunk['chunk_id']
        
        return sample

        
    
        

        
            
            
        