# model_loader.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model  # LoRA 어댑터 결합을 위한 임포트

class UniversalModelLoader:
    """HuggingFace 등 가중치 아티팩트를 가져와 Config 설정 모드에 맞게 그래프를 변형하는 결합 레이어"""
    def __init__(self, config):
        self.config = config
        self.model_id = config.MODEL_ID
        self.setting = config.SETTING
        self.device = config.DEVICE  # config에서 cuda:0 혹은 에이전트 할당량 참조

    def load_backbone_and_tokenizer(self):
        print(f"📥 [하드웨어 진입] 오리지널 레포지토리 [{self.model_id}] 실물 가중치 다운로드 및 GPU 배치 시작...")
        
        # 실물 transformers 인스턴스 가동 및 프로세서(토크나이저 포함) 로드
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        # 모델을 로드하면서 지정된 GPU 단일 디바이스로 강제 할당 (fp16 캐스팅으로 메모리 절약)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            device_map=self.device     # nvidia-smi에 VRAM이 차오르게 만드는 핵심 스위치
        )
        
        return model, processor

    # 💡 에러 해결 지점: 누락되었던 실험 설정(PEFT / Full fine-tuning 등) 적용 메소드 완벽 구현
    def apply_experiment_setting(self, model):
        """Zero-shot, Few-shot, Full fine-tuning, PEFT 세팅에 맞추어 그라디언트 및 어댑터를 자동 구성"""
        print(f"🛠️ [전략 분기 변형] 현재 활성화된 세팅 모드: {self.setting}")
        
        if self.setting == "Zero-shot":
            print(" -> [Freeze 전역] 가중치 업데이트 전면 차단 (Inference 전용 모드)")
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return {"model": model, "trainable": False, "adapter": None}
            
        elif self.setting == "Few-shot":
            print(" -> [Context 전용] 고정 가중치 유지 및 프롬프트 인컨텍스트 예시 주입 버퍼 할당")
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return {"model": model, "trainable": False, "adapter": "InContextBuffer"}
            
        elif self.setting == "Full fine-tuning":
            print(" -> [Unfreeze 전역] 모든 계층 역전파 가동 (대규모 컴퓨테이션 연산 점유)")
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            return {"model": model, "trainable": True, "adapter": None}
            
        elif self.setting == "PEFT":
            print(" -> [Parameter-Efficient] 백본 동결 및 LoRA 어댑터 타겟 계층 동적 결합 매핑 가동")
            
            # Qwen2-VL 아키텍처의 대표적인 시각/언어 프로젝션 선형 레이어를 타겟으로 지정합니다.
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Attention 계층 타겟팅
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            # 순수 백본 모델에 LoRA 파라미터 그래프 주입 및 래핑
            peft_model = get_peft_model(model, peft_config)
            peft_model.print_trainable_parameters() # 학습 가능 파라미터 수 통계 출력
            
            return {"model": peft_model, "trainable": "AdapterOnly", "adapter": "LoRA_Weights_v1.0"}
            
        else:
            raise ValueError(f"정의되지 않은 아키텍처 아규먼트 설정: {self.setting}")