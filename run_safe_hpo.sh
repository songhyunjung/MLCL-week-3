#!/bin/bash

# 1. 아나콘다 가상환경 강제 활성화
source /NasData2/home/shj/anaconda3/etc/profile.d/conda.sh
conda activate week3

# 2. 하이퍼파라미터 조합 정의 (총 8개 조합)
MODELS=("Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen3-VL-8B-Instruct")
LRS=("2e-5" "5e-5")
DECODINGS=("Greedy" "Beam")

# 💡 실시간으로 완전히 비어있는(VRAM 사용량 500MB 이하) GPU ID를 찾아주는 함수
get_clean_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk '{if ($2 < 500) print $1}' | head -n 1
}

echo "🛡️ 안전 모드 하이퍼파라미터 오토메이션 스크립트 가동"
echo "----------------------------------------------------"

JOB_COUNT=1

for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        for DEC in "${DECODINGS[@]}"; do
            
            # 1. 현재 완전히 비어있는 GPU가 생길 때까지 1분 간격으로 무한 대기 (선배님들 자원 보호)
            while true; do
                TARGET_GPU=$(get_clean_gpu)
                
                if [ ! -z "$TARGET_GPU" ]; then
                    break  # 비어있는 GPU를 찾았다면 루프 탈출
                fi
                
                echo "⏳ [대기] 현재 모든 GPU를 다른 연구원이 사용 중입니다. 안전을 위해 1분간 대기합니다..."
                sleep 60
            done
            
            CLEAN_NAME=$(echo $MODEL | awk -F'/' '{print $2}')
            echo "[실험 $JOB_COUNT/8] $CLEAN_NAME | LR: $LR | $DEC ──> 🔒 안전하게 비어있는 [GPU $TARGET_GPU] 확보 완료!"
            
            # 2. 해당 GPU만 보이도록 완전 격리 후 백그라운드 실행
            export CUDA_VISIBLE_DEVICES=$TARGET_GPU
            
            # 독립된 프로세스로 실행하며 로그는 각각의 파일로 분리 저장
            nohup python main_vlm.py \
                --model_id "$MODEL" \
                --lr "$LR" \
                --decoding "$DEC" \
                --gpu_id 0 > "log_${CLEAN_NAME}_LR${LR}_${DEC}.txt" 2>&1 &
            
            # 프로세스가 메모리를 완전히 점유할 시간을 주기 위한 안전 마진 (15초 대기)
            sleep 15
            
            JOB_COUNT=$((JOB_COUNT + 1))
            
            # 한 번에 GPU 4개를 다 채웠다면, 먼저 들어간 실험들이 끝날 때까지 안전하게 대기
            if [ $((JOB_COUNT % 4)) -eq 0 ]; then
                echo "⏳ GPU 배치 가동 중... 실행 중인 작업들이 완전히 끝날 때까지 대기(wait)합니다."
                wait
                echo "✅ 이전 배치가 안전하게 완료되었습니다. 다음 조합으로 진입합니다."
            fi

        done
    done
done

wait
echo "🎉 다른 사람의 자원을 전혀 건드리지 않고, 8개 실험을 완료했습니다! ClearML 웹 UI를 확인하세요."
