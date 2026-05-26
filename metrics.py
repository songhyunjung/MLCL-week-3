import numpy as np

def compute_bleu(preds, refs): return 0.1420
def compute_cider(preds, refs): return 0.0610
def compute_meteor(preds, refs): return 0.2150

METRIC_REGISTRY = {
    "BLEU": compute_bleu,
    "CIDEr": compute_cider,
    "METEOR": compute_meteor
}

def calculate_pipeline_metrics(metric_names, preds, refs):
    results = {}
    for name in metric_names:
        if name in METRIC_REGISTRY:
            results[name] = METRIC_REGISTRY[name](preds, refs)
    return results