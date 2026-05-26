# metrics.py (실제 pycocoevalcap 연동 및 토큰화 버그 수정 버전)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def prepare_coco_format(preds, refs):
    """
    일반적인 파이썬 리스트 형태를 COCO Eval 라이브러리 포맷으로 변환합니다.
    - preds: ["예측문장1", "예측문장2", ...]
    - refs: [["정답1-1", "정답1-2"], ["정답2-1"], ...] 형태 혹은 ["정답1", "정답2", ...] 형태 모두 대응
    """
    gts_dict = {}
    res_dict = {}
    
    for idx, (pred, ref) in enumerate(zip(preds, refs)):
        # 1. 예측값 변환 (샘플당 1개의 문장)
        res_dict[idx] = [{"caption": str(pred).strip()}]
        
        # 2. 정답(참조값) 변환 (샘플당 여러 개의 다중 정답 후보를 가질 수 있음)
        if isinstance(ref, list):
            gts_dict[idx] = [{"caption": str(r).strip()} for r in ref]
        else:
            gts_dict[idx] = [{"caption": str(ref).strip()}]
            
    return gts_dict, res_dict

def compute_coco_metrics(preds, refs):
    """
    예측값과 정답값을 COCO 포맷으로 변환 후, 
    PTBTokenizer를 통해 완벽하게 토큰화하여 BLEU, CIDEr, METEOR 등을 계산합니다.
    """
    if not preds or not refs:
        return {}

    # 1. 기본 COCO 사전 포맷으로 변환
    gts_dict, res_dict = prepare_coco_format(preds, refs)
    
    # 2. 🚨 [버그 해결] PTBTokenizer를 적용해 텍스트를 형태소/단어 단위로 쪼개줍니다.
    # 이 단계를 거쳐야 Bleu와 Cider의 내부 사전(Vocabulary) 매핑이 일치하여 0점이 나오지 않습니다.
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_dict)
    res = tokenizer.tokenize(res_dict)
    
    # 3. 평가지표 계산기 객체 정의
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    
    final_scores = {}
    
    # 4. 루프를 돌며 진짜 연산 수행
    for scorer, metric_name in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(metric_name, list):
                # BLEU 점수는 1, 2, 3, 4차 점수가 리스트로 반환됩니다.
                for n, metric in enumerate(metric_name):
                    final_scores[metric] = score[n]
            else:
                final_scores[metric_name] = score
        except Exception as e:
            print(f"⚠️ [{metric_name}] 계산 중 오류 발생: {e}")
            if isinstance(metric_name, list):
                for metric in metric_name:
                    final_scores[metric] = 0.0
            else:
                final_scores[metric_name] = 0.0
            
    return final_scores

# --- 기존 하위 호환성 유지를 위한 래퍼 함수들 ---

def compute_bleu(preds, refs):
    scores = compute_coco_metrics(preds, refs)
    return scores.get("BLEU_4", 0.0)

def compute_cider(preds, refs):
    scores = compute_coco_metrics(preds, refs)
    return scores.get("CIDEr", 0.0)

def compute_meteor(preds, refs):
    scores = compute_coco_metrics(preds, refs)
    return scores.get("METEOR", 0.0)

METRIC_REGISTRY = {
    "BLEU": compute_bleu,
    "CIDEr": compute_cider,
    "METEOR": compute_meteor
}

def calculate_pipeline_metrics(metric_names, preds, refs):
    """
    외부 스크립트(main_vlm.py 등)에서 기존 방식 그대로 호출할 수 있게 설계된 메인 파이프라인 함수
    """
    # 원본 함수인 compute_coco_metrics를 수행하여 모든 지표를 일괄 계산한 뒤 필터링합니다.
    all_scores = compute_coco_metrics(preds, refs)
    
    filtered_scores = {}
    for name in metric_names:
        if name == "BLEU":
            filtered_scores["BLEU_4"] = all_scores.get("BLEU_4", 0.0)
        else:
            filtered_scores[name] = all_scores.get(name, 0.0)
            
    return filtered_scores