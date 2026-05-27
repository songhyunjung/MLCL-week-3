import torch

class AppConfig:
    # 1. 인프라 및 예방용 초고속 디버그 검증 플래그
    GPU_ID = 0
    DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    DEBUG_MODE = False  # True일 시 전체 데이터 중 10개만 추출하여 전체 파이프라인 결함 선제 검증
    
    # 2. MLOps 및 코드 형상 관리를 위한 중앙 원격 Git 제어 설정
    PROJECT_NAME = "Week3_Qwen2"
    GIT_REPOSITORY = "https://github.com/songhyunjung/MLCL-week-3.git"
    GIT_BRANCH = "main"
    
    # 3. [연구자 제어판] 여기만 수정하면 전체 하위 컴포넌트 로직이 자동 변경됨

    # 백본 모델 경로, Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct 중 선택
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  

    # Zero-shot, Few-shot, Full fine-tuning, PEFT
    SETTING = "PEFT"          

    # Greedy, Beam (Beam=5) 중 선택     
    DECODING_METHOD = "Beam"    #

    # Flickr30k, CustomCOCO 등 확장 가능            
    DATASET_NAME = "nlphuji/flickr30k"    

    # 4. 하이퍼파라미터 컴포넌트
    BATCH_SIZE = 2
    EPOCHS = 3
    # 2e-5, 5e-5 중에 선택
    LEARNING_RATE = 2e-5 #
    MAX_NEW_TOKENS = 40
    
    # 5. 평가지표 동적 활성화 풀
    BEAM_SIZE = 5
    ACTIVE_METRICS = ["BLEU", "CIDEr", "METEOR"]