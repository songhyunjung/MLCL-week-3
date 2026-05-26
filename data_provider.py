# data_provider.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

class UniversalDataProvider:
    """어떤 유형의 데이터셋이 들어와도 Config 기반으로 파싱하여 공급하는 범용 데이터 레이어"""
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.DATASET_NAME  # "nlphuji/flickr30k"
        self.debug_mode = config.DEBUG_MODE
        
    def load_dataset(self):
        print(f"📦 [{self.dataset_name}] HuggingFace 원천 데이터셋 라이브 데이터 로드 시작...")
        
        try:
            # nlphuji/flickr30k 데이터셋의 'test' 혹은 'train' 스플릿 로드 (여기서는 테스트용으로 'test' 세그먼트 로드 예시)
            # 대용량인 경우 split="train" 등으로 조절 가능합니다.
            hf_dataset = load_dataset(self.dataset_name, split="test")
            
            # HuggingFace Dataset 객체를 파이프라인 호환을 위해 Pandas DataFrame으로 캐스팅
            df = pd.DataFrame(hf_dataset)
            
            # flickr30k 데이터셋 특성에 맞춰 메타데이터 컬럼명 보정 작업 (필요시 활성화)
            # 기본적으로 'image', 'caption' 컬럼이 존재합니다.
            if 'split' not in df.columns:
                df['split'] = 'test'
                
        except Exception as e:
            print(f"⚠️ [Data Engine] 실물 데이터셋 로드 실패 ({e}). 에뮬레이션 패크백 데이터를 생성합니다.")
            mock_data = [
                {"image_id": "img1.jpg", "caption": ["A black dog jumping over a hurdle."], "split": "test"},
                {"image_id": "img2.jpg", "caption": ["Two people walk on a busy sidewalk."], "split": "test"}
            ] * 20
            df = pd.DataFrame(mock_data)
        
        # ⚠️ [안전장치] 디버그 모드일 경우 대규모 연산 전에 초고속으로 코드가 깨지는지 선제 검증
        if self.debug_mode:
            print("🚨 [data_provider] DEBUG_MODE 감지: 데이터 세트를 초소형 크기(10개)로 강제 제한합니다.")
            df = df.head(10)
            
        print(f"✅ 데이터 로드 완료! 총 로우 수: {len(df)}")
        return df

    def generate_eda_report(self, df, output_path="eda_distribution.png"):
        """데이터셋의 데이터 분할 분포 통계를 확인하고 시각화 보고서를 자동 빌드하는 공통 모듈"""
        print("📊 데이터셋 통계 정보 분석 및 EDA 시각화 레포트 빌드 중...")
        if 'split' in df.columns:
            df['split'].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"Dataset Split Distribution - {self.dataset_name}")
            plt.xlabel("Split")
            plt.ylabel("Count")
            plt.savefig(output_path)
            plt.close()
            print(f" 💾 EDA 시각화 레포트 저장 완료: {output_path}")