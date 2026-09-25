"""
Evaluation metric module implementing macro-averaged F0.5 score as specified by the Amazon ML Challenge 2026.
"""

from typing import Dict, Set, Tuple

def compute_entity_metrics(pred_set: Set[str], gt_set: Set[str], beta: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute entity-level Precision, Recall, and F_beta score for a single Source 1 entity.
    
    Rules for singletons (gt_set is empty):
    - Correctly predicting empty set (pred_set empty) -> Precision=1.0, Recall=1.0, F_beta=1.0
    - Predicting false matches (pred_set non-empty) -> Precision=0.0, Recall=0.0, F_beta=0.0
    """
    pred_set = set(pred_set) if isinstance(pred_set, (list, set, tuple)) else set()
    gt_set = set(gt_set) if isinstance(gt_set, (list, set, tuple)) else set()

    if not gt_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
            
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = beta ** 2
    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)

    return precision, recall, f_beta

def evaluate_predictions(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]], beta: float = 0.5) -> dict:
    """
    Compute macro-averaged Precision, Recall, and F_0.5 scores across all S1 entities in ground_truth.
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f05 = 0.0
    count = len(ground_truth)

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f05 = compute_entity_metrics(pred_set, gt_set, beta=beta)
        total_precision += p
        total_recall += r
        total_f05 += f05

    macro_precision = total_precision / count if count > 0 else 0.0
    macro_recall = total_recall / count if count > 0 else 0.0
    macro_f05 = total_f05 / count if count > 0 else 0.0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f0.5": macro_f05,
        "macro_f05": macro_f05,
        "exact_singletons_credit": 1.0,
        "total_evaluated_s1": count,
    }

def run_evaluation_unit_tests():
    """Unit test suite verifying macro F0.5 calculation against manual calculations."""
    print("Running evaluation unit tests...")
    
    # Example 1: Perfect prediction
    p, r, f = compute_entity_metrics({'S2-1', 'S3-1'}, {'S2-1', 'S3-1'})
    assert p == 1.0 and r == 1.0 and f == 1.0, f"Failed perfect match: p={p}, r={r}, f={f}"

    # Example 2: Precision penalty (Example from README.md)
    p, r, f = compute_entity_metrics({'S2-00047', 'S2-00193', 'S3-00812'}, {'S2-00047', 'S3-00812'})
    assert round(p, 3) == 0.667, f"Precision failed: {p}"
    assert r == 1.0, f"Recall failed: {r}"
    assert round(f, 3) == 0.714, f"F0.5 failed: {f} (expected 0.714)"

    # Example 3: Correct singleton
    p, r, f = compute_entity_metrics(set(), set())
    assert p == 1.0 and r == 1.0 and f == 1.0, "Singleton empty match failed"

    # Example 4: False match on singleton
    p, r, f = compute_entity_metrics({'S2-100'}, set())
    assert p == 0.0 and r == 0.0 and f == 0.0, "Singleton false match failed"

    print("ALL EVALUATION UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_evaluation_unit_tests()
