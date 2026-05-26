# hpo.py
from clearml import Task
# 💡 GridSearch를 상단에서 직접 임포트해 줍니다.
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from config import AppConfig

# 1. HPO 마스터 태스크 초기화
task = Task.init(
    project_name="MLCL_Week3_VLM", 
    task_name=f"{AppConfig.MODEL_ID.split('/')[-1]}_{AppConfig.SETTING}",
    task_type=Task.TaskTypes.optimizer
)

# 2. 하이퍼파라미터 오토메이션 엔진 설정
optimizer = HyperParameterOptimizer(
    "ddc6dd590490437da7c721afd8e80d80",
    
    hyper_parameters=[
        DiscreteParameterRange('Args/LEARNING_RATE', values=[2e-5, 5e-5]),
        DiscreteParameterRange('Args/DECODING_METHOD', values=['Greedy', 'Beam']),
        
        DiscreteParameterRange('Args/SETTING', values=['PEFT']),
        DiscreteParameterRange('Args/EPOCHS', values=[3]),
        DiscreteParameterRange('Args/BATCH_SIZE', values=[2]),
        DiscreteParameterRange('Args/BEAM_SIZE', values=[5]),
    ],
    
    objective_metric_title='Evaluation Metrics',
    objective_metric_series='CIDEr',
    objective_metric_sign='max',
    
    # 💡 에러 지점 수정: 앞의 클래스 껍데기를 떼고 GridSearch를 직접 바인딩합니다.
    optimizer_class=GridSearch,
    max_number_of_concurrent_tasks=4,  # 4대 GPU 동시 가동
    total_max_jobs=4                   # 딱 4개 조합 탐색 후 종료
)

# HPO 스케줄러 가동
optimizer.start()
optimizer.wait()
optimizer.stop()

print("🎉 4개 조합 자동 실험이 성공적으로 인큐잉되었습니다.")