# hpo.py
from clearml import Task
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from config import AppConfig

# 1. HPO 마스터 태스크 초기화
task = Task.init(
    project_name="MLCL_Week3_VLM", 
    task_name=f"{AppConfig.MODEL_ID.split('/')[-1]}_{AppConfig.SETTING}_Optimizer",
    task_type=Task.TaskTypes.optimizer
)

# 🔥 [핵심 자동화] 프로젝트에서 가장 최근에 등록된 원본(Base) 태스크를 동적으로 찾습니다.
# 매번 수동으로 ID를 복사해서 붙여넣을 필요가 없어집니다!
try:
    base_tasks = Task.get_tasks(
        project_name=AppConfig.PROJECT_NAME,
        task_name=f"{AppConfig.MODEL_ID.split('/')[-1]}_{AppConfig.SETTING}",
        task_filter={
            "status": ["completed", "created", "queued"], # 살아있거나 성공한 태스크 타겟
            "type": ["training"]                         # 마스터 오토메이터 제외
        }
    )
    # 가장 최근에 생성된 태스크의 ID를 자동으로 추출
    dynamic_base_task_id = base_tasks[0].id
    print(f"🎯 [자동 연동 성공] 최신 베이스 태스크 ID 발견: {dynamic_base_task_id}")
except Exception as e:
    # 혹시나 서버에 태스크가 하나도 없을 때를 대비한 안전 예외처리
    raise ValueError("⚠️ ClearML 서버에서 베이스 템플릿 태스크를 찾을 수 없습니다. main_vlm.py를 로컬에서 최소 1회 실행해 주세요.")

# 2. 하이퍼파라미터 오토메이션 엔진 설정
optimizer = HyperParameterOptimizer(
    # ⚠️ 고정된 문자열 ID 대신 자동으로 찾아온 dynamic_base_task_id를 넣어줍니다!
    base_task_id=dynamic_base_task_id,
    
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
    optimizer_class=GridSearch,
    execution_queue='default',
    
    # ⚠️ 에이전트가 패치 충돌(git diff)을 일으키지 않도록 완전 동기화 옵션 강제 유도
    spawn_project=None,
    save_top_k=3
)

# 3. 최적화 루프 가동
optimizer.start()
optimizer.wait()
optimizer.stop()