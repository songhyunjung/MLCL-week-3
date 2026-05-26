# Week 3: VLM Image Captioning Experiment

이 프로젝트는 시각-언어 모델(VLM)인 **Qwen2-VL** 및 **Qwen3-VL**을 활용하여 이미지 캡셔닝(Image Captioning) 성능을 실험하고, PEFT(LoRA) 기법을 통해 효율적으로 미세조정(Fine-tuning)하는 3주차 통합 연구 저장소입니다.

---

## 1. 주요 기능 및 파이프라인

* **VLM 기반 이미지 캡셔닝**: Qwen2-VL / Qwen3-VL 모델을 활용한 고성능 이미지 설명 생성
* **PEFT (LoRA) 미세조정**: `transformers`, `peft`, `accelerate`를 활용한 효율적인 모델 학습
* **실시간 정성평가 데모**: `Gradio`를 활용한 웹 UI 기반의 이미지 캡셔닝 데모 구동
* **MLOps 실험 관리**: `ClearML`을 연동한 학습 프로세스, 하이퍼파라미터 및 Loss 메트릭 실시간 모니터링
* **데이터 시각화**: `Matplotlib`, `Seaborn`, `Plotly`를 활용한 데이터셋 분포 및 실험 결과 분석

---

## 2. 개발 환경 및 요구 사항

* **OS**: Linux / Ubuntu (추천)
* **Python**: `>=3.10, <3.12`
* **CUDA**: `12.2` (또는 12.x 그래픽 드라이버 환경)
* **패키지 관리자**: `Poetry`

---

## 3. 시작하기 (Installation & Setup)

### 가상환경 활성화 및 의존성 설치
본 프로젝트는 CUDA 12.1 기반으로 빌드된 PyTorch 2.x 버전을 강제하도록 설정되어 있습니다.

```bash
# 1. Conda 가상환경 생성 및 활성화 (Python 3.11 권장)
conda create -n vlm-env python=3.11 -y
conda activate vlm-env

# 2. Poetry가 현재 Conda 환경을 바라보도록 설정
poetry config virtualenvs.create false

# 3. 의존성 라이브러리 일괄 설치
poetry install