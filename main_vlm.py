import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from clearml import Task

from config import AppConfig
from data_provider import UniversalDataProvider
from model_loader import UniversalModelLoader
from metrics import calculate_pipeline_metrics
import argparse

# =================================================================
# ⚙️ 실전 학습을 위한 파이토치 Dataset 래퍼 클래스 정의
# =================================================================
class VLMDataset(Dataset):
    """Pandas DataFrame 구조의 텍스트와 프로세서를 VLM 학습 포맷으로 바인딩합니다."""
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 정답 캡션 텍스트 추출 (리스트 형태 대응)
        caption = row['caption']
        if isinstance(caption, list):
            caption_text = str(caption[0])
        else:
            caption_text = str(caption)
            
        # 2. Qwen2-VL용 멀티모달 프롬프트 구성 (텍스트 기반 우선 구현)
        # 이미지 데이터가 PIL 인스턴스로 'image' 컬럼에 있을 경우 결합 가능
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption_text}]
            }
        ]
        
        return {"messages": messages, "raw_caption": caption_text}

def collate_fn_with_processor(batch, processor, device):
    """배치 데이터를 Qwen2-VL 전용 입력을 위한 텐서 크기로 템플릿 패딩 처리를 수행합니다."""
    texts = [processor.apply_chat_template(sample["messages"], tokenize=False) for sample in batch]
    raw_captions = [sample["raw_caption"] for sample in batch]
    
    # 모델 포워딩을 위한 패딩 및 텐서 정렬 생성
    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt"
    )
    
    # 💡 Causal LM 구조의 학습을 위해 input_ids와 동일하게 labels를 주입합니다.
    inputs["labels"] = inputs["input_ids"].clone()
    
    # 지정된 GPU 디바이스로 물리 배치 이동
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    return inputs, raw_captions


# =================================================================
# 🚀 메인 파이프라인 가동 함수
# =================================================================
def main():
    print("==================================================")
    print("🚀 범용 전역 파이프라인 가동 (Qwen2-VL 실전 학습 및 리얼 데이터 평가 버전)")
    print("==================================================")
    
    # 1. ClearML 실험 세션 초기화 및 Git 원격 형상 강제 바인딩
    task = Task.init(
        project_name=AppConfig.PROJECT_NAME, 
        task_name=f"{AppConfig.DECODING_METHOD}_{AppConfig.LEARNING_RATE}",
        reuse_last_task_id=False
    )
    task.set_repo(repo=AppConfig.GIT_REPOSITORY, branch=AppConfig.GIT_BRANCH, commit=None)
    logger = task.get_logger()
    
    # HPO 핵심 파라미터 링크 인터페이스 연동 (WARNING 근본 해결)
    args_dict = {
        'LEARNING_RATE': AppConfig.LEARNING_RATE,
        'DECODING_METHOD': AppConfig.DECODING_METHOD,
        'SETTING': AppConfig.SETTING,
        'EPOCHS': AppConfig.EPOCHS,
        'BATCH_SIZE': AppConfig.BATCH_SIZE,
        'BEAM_SIZE': AppConfig.BEAM_SIZE,
    }
    task.connect(args_dict, name='Args') 
    
    # 하이퍼파라미터 주입 동기화
    AppConfig.LEARNING_RATE = args_dict['LEARNING_RATE']
    AppConfig.DECODING_METHOD = args_dict['DECODING_METHOD']
    AppConfig.SETTING = args_dict['SETTING']
    AppConfig.EPOCHS = args_dict['EPOCHS']
    AppConfig.BATCH_SIZE = args_dict['BATCH_SIZE']
    AppConfig.BEAM_SIZE = int(args_dict['BEAM_SIZE'])
    
    print(f"⚙️ [ClearML Injection] 최적화 하이퍼파라미터 주입 완료:")
    print(f"  - LEARNING_RATE: {AppConfig.LEARNING_RATE} | DECODING: {AppConfig.DECODING_METHOD}")
    print(f"  - SETTING: {AppConfig.SETTING} | EPOCHS: {AppConfig.EPOCHS} | BATCH: {AppConfig.BATCH_SIZE}")

    # 2. 범용 데이터 및 모델 인프라 컴포넌트 로드
    data_engine = UniversalDataProvider(AppConfig)
    dataset_df = data_engine.load_dataset()
    data_engine.generate_eda_report(dataset_df)
    
    model_engine = UniversalModelLoader(AppConfig)
    raw_model, processor = model_engine.load_backbone_and_tokenizer()
    configured_model_pack = model_engine.apply_experiment_setting(raw_model)
    
    # 알맹이 모델 추출
    vlm_model = configured_model_pack["model"]
    target_device = AppConfig.DEVICE
    print(f"🖥️ 실물 가중치 바인딩 하드웨어 디바이스: {target_device}")

    # 3. 데이터셋 분할 및 실전 파이토치 DataLoader 구성
    # 샘플 코드로 전면 데이터를 학습/검증으로 쪼개어 사용합니다.
    train_df = dataset_df.sample(frac=0.8, random_state=42)
    val_df = dataset_df.drop(train_df.index)
    
    train_dataset = VLMDataset(train_df, processor)
    val_dataset = VLMDataset(val_df, processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(AppConfig.BATCH_SIZE), 
        shuffle=True,
        collate_fn=lambda b: collate_fn_with_processor(b, processor, target_device)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=int(AppConfig.BATCH_SIZE), 
        shuffle=False,
        collate_fn=lambda b: collate_fn_with_processor(b, processor, target_device)
    )
    
    # 4. [실전 가동 Core] 학습 세팅 분기
    if AppConfig.SETTING in ["Full fine-tuning", "PEFT"]:
        print(f"🔥 [{AppConfig.SETTING}] 모드 활성화: Qwen2-VL 가중치 기반 실전 그라디언트 루프 가동")
        
        # 가짜 nn.Linear 최적화 대신 실제 VLM 파라미터를 옵티마이저에 연동
        optimizer = optim.AdamW(vlm_model.parameters(), lr=AppConfig.LEARNING_RATE)
        
        for epoch in range(int(AppConfig.EPOCHS)):
            print(f"🌀 Current Progress Train Loop - Epoch [{epoch + 1}/{AppConfig.EPOCHS}]")
            
            # ----------------------------------------------------------------
            # 훈련 페이즈 (실제 Qwen2-VL 모델 학습)
            # ----------------------------------------------------------------
            vlm_model.train()
            total_train_loss = 0.0
            steps = 0
            
            for batch_inputs, _ in train_loader:
                # MLOps 파이프라인 흐름상 1에폭당 최대 10스텝만 시범 가동하여 조기 병목 방지 제어
                if steps >= 10:
                    break
                    
                optimizer.zero_grad()
                
                # 🚨 [액션 플랜 반영] 가짜 난수가 아닌 실물 데이터 배치를 모델에 주입하여 forward 연산
                outputs = vlm_model(
                    input_ids=batch_inputs.get("input_ids"),
                    attention_mask=batch_inputs.get("attention_mask"),
                    labels=batch_inputs.get("labels")
                )
                
                # Qwen2-VL이 자체 반환한 실제 CrossEntropyLoss 추출
                loss = outputs.loss
                
                # 역전파 및 가중치 업데이트 연산
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                steps += 1
                
            avg_train_loss = total_train_loss / max(1, steps)
            
            # 💡 [ClearML 연동] 실제 계산된 train_loss 대시보드 리포팅
            logger.report_scalar(
                title="Loss", series="train_loss", value=avg_train_loss, iteration=epoch + 1
            )
            
            # ----------------------------------------------------------------
            # 검증 페이즈 (Inference 및 리얼 텍스트 평가지표 계산)
            # ----------------------------------------------------------------
            vlm_model.eval()
            total_val_loss = 0.0
            val_steps = 0
            
            all_generated_preds = []
            all_ground_truths = []
            
            with torch.no_grad():
                for batch_inputs, raw_captions in val_loader:
                    if val_steps >= 5: # 검증 스텝 제한 제어
                        break
                        
                    # 검증 Loss 계산
                    outputs = vlm_model(
                        input_ids=batch_inputs.get("input_ids"),
                        attention_mask=batch_inputs.get("attention_mask"),
                        labels=batch_inputs.get("labels")
                    )
                    total_val_loss += outputs.loss.item()
                    
                    # 🚨 [액션 플랜 반영] 하드코딩 문장을 완전히 지우고 실제 model.generate() 추론 수행
                    # 디코딩 방식 정의 (Beam vs Greedy)
                    gen_kwargs = {"max_new_tokens": 30, "pad_token_id": processor.tokenizer.pad_token_id}
                    if AppConfig.DECODING_METHOD == "Beam":
                        gen_kwargs["num_beams"] = AppConfig.BEAM_SIZE
                        gen_kwargs["early_stopping"] = True
                        
                    generated_ids = vlm_model.generate(
                        input_ids=batch_inputs.get("input_ids"),
                        attention_mask=batch_inputs.get("attention_mask"),
                        **gen_kwargs
                    )
                    
                    # 생성된 인덱스를 실제 영문 텍스트로 복원 디코딩
                    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    all_generated_preds.extend(preds)
                    all_ground_truths.extend(raw_captions)
                    val_steps += 1
                    
            avg_val_loss = total_val_loss / max(1, val_steps)
            
            # 💡 [ClearML 연동] 실제 계산된 val_loss 대시보드 리포팅
            logger.report_scalar(
                title="Loss", series="val_loss", value=avg_val_loss, iteration=epoch + 1
            )
            
            # 🚨 [액션 플랜 반영] 가짜 보정 상수(boost_factor) 사기 연산 완전 삭제!
            # 진짜 생성된 문장(all_generated_preds)과 실제 정답(all_ground_truths)으로 평가 지표 계산
            epoch_metrics = calculate_pipeline_metrics(AppConfig.ACTIVE_METRICS, all_generated_preds, all_ground_truths)
            
            # 순수 정량 평가 점수만 리포팅
            for m_name, s_value in epoch_metrics.items():
                logger.report_scalar(
                    title="Evaluation Metrics", 
                    series=m_name, 
                    value=float(s_value), 
                    iteration=epoch + 1
                )
            
            print(f"  ✅ Epoch [{epoch + 1}] 평가지표 계산 완료 | 리얼 Train Loss: {avg_train_loss:.4f} | 리얼 Val Loss: {avg_val_loss:.4f}")
            
    else:
        print(f"❄️ [{AppConfig.SETTING}] 모드 가동: 추론 전용 패스이므로 백엔드 대규모 학습 연산을 건너뜁니다.")

    print("==================================================")
    print("🎉 전역 학습 파이프라인 루프가 성공적으로 종료되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    # 인자 생성기 등록
    parser = argparse.ArgumentParser(description="VLM Hyperparameter Runner")
    parser.add_argument("--model_id", type=str, default=AppConfig.MODEL_ID)
    parser.add_argument("--lr", type=float, default=AppConfig.LEARNING_RATE)
    parser.add_argument("--decoding", type=str, default=AppConfig.DECODING_METHOD)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # 1. 전달받은 인자로 인프라 및 실험 셋업 동적 업데이트
    AppConfig.GPU_ID = args.gpu_id
    import torch # GPU_ID 세팅 후 디바이스 빌드를 위해 토치 재체크 유도 가능
    AppConfig.DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    AppConfig.update_config(args.model_id, args.lr, args.decoding)

    # 2. ClearML 태스크 이름도 고유하게 생성되도록 변경
    clean_model_name = AppConfig.MODEL_ID.split('/')[-1]
    task = Task.init(
        project_name=AppConfig.PROJECT_NAME, 
        task_name=f"[{clean_model_name}]_LR_{AppConfig.LEARNING_RATE}_DEC_{AppConfig.DECODING_METHOD}"
    )
    
    # --- 이하 기존 main_vlm.py 본문 코드 유지 ---
    print(f"🚀 활성화된 하드웨어 디바이스: {AppConfig.DEVICE}")
    logger = task.get_logger()
    main()