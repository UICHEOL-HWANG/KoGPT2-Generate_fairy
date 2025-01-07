import argparse
import os

from torch.utils.data import DataLoader
from datasets import load_dataset

from src.KoGPTDatasets import KoGPTDataset, split_text_into_chunks_with_metadata
from src.model_manager import ModelManager
from src.training_manager import Training_Manager
from src.seperation_data import seperated_data


def parse_args():
    """
    학습을 위한 명령줄 파라미터 설정
    """
    parser = argparse.ArgumentParser(description='KoGPT2 Fine-Tuning')

    parser.add_argument('--model_save_path', type=str, default='./results/kogpt2-finetuned',
                        help='모델 저장 경로')
    parser.add_argument('--max_length', type=int, default=512,
                        help='시퀀스 최대 길이')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='배치 사이즈')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='에포크 수')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='학습률')

    return parser.parse_args()


def main():
    args = parse_args()

    # ✅ 모델 저장 디렉터리 생성
    os.makedirs(args.model_save_path, exist_ok=True)
    print('✅ 모델 디렉토리 생성 성공!')

    # ✅ Hugging Face 데이터셋 로드
    dataset = load_dataset("UICHEOL-HWANG/fairy_dataset")
    tmp_data = dataset['train']

    # ✅ ModelManager 초기화
    model_manager = ModelManager(learning_rate=args.learning_rate, epochs=args.num_epochs)
    tokenizer = model_manager.tokenizer

    # ✅ 데이터 분할
    print("✅ 데이터 분할 중...")
    train_data, val_data = seperated_data(tmp_data)
    train_chunks = split_text_into_chunks_with_metadata(train_data, tokenizer, max_length=args.max_length, overlap=50)
    val_chunks = split_text_into_chunks_with_metadata(val_data, tokenizer, max_length=args.max_length, overlap=50)

    train_dataset = KoGPTDataset(train_chunks, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = KoGPTDataset(val_chunks, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    print("✅ 데이터 로드 완료. 훈련 시작...")

    # ✅ Training Manager 초기화 및 학습
    trainer = Training_Manager(model_manager.model, train_loader, val_loader)

    print("🚀 모델 학습 시작...")
    train_losses = trainer.train(args.num_epochs)

    if train_losses:
        print(f"✅ 모든 에포크 완료! 총 {len(train_losses)}개의 에포크가 학습되었습니다.")
    else:
        print("❌ `train_losses`가 비어있습니다. `Training_Manager.train()`을 확인하세요.")

    # ✅ 검증
    print("🔄 모델 검증 시작...")
    val_loss, val_accuracy = trainer.validation()
    print(f"✅ 훈련 완료! 최종 검증 손실: {val_loss:.4f}, 정확도: {val_accuracy:.4f}")

    # ✅ 모델 및 토크나이저 저장
    print("💾 모델과 토크나이저 저장 중...")
    model_manager.model.save_pretrained(args.model_save_path)
    model_manager.tokenizer.save_pretrained(args.model_save_path)
    print(f"✅ 모델과 토크나이저가 {args.model_save_path}에 저장되었습니다.")

    # ✅ Hugging Face 허브에 업로드
    print("🚀 Hugging Face 허브에 모델 업로드 시작...")
    model_manager.model.push_to_hub("UICHEOL-HWANG/KoGPT2-fairytail")
    model_manager.tokenizer.push_to_hub("UICHEOL-HWANG/KoGPT2-fairytail")
    print("✅ 모델과 토크나이저가 Hugging Face 허브에 성공적으로 업로드되었습니다.")


if __name__ == '__main__':
    main()
