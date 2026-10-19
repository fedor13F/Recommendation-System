from typing import List, Dict
import numpy as np


def recall_at_k(predicted_items: List[int], true_items: List[int], k: int) -> float:
    if len(true_items) == 0:
        return 0.0

    predicted_items = predicted_items[:k]
    count_rel_in_top = len(set(predicted_items) & set(true_items))

    return count_rel_in_top / len(true_items)


def hit_rate_at_k(predicted_items: List[int], true_items: List[int], k: int) -> int:
    return int(len(set(predicted_items[:k]) & set(true_items)) > 0)


def rr_at_k(predicted_items: List[int], true_items: List[int], k: int) -> float:
    if len(true_items) == 0:
        return 0.0
    true_items = set(true_items)
    for i in range(len(predicted_items[:k])):
        if predicted_items[i] in true_items:
            return 1 / (i + 1)
    return 0.0


def precision_at_k(predicted_items: List[int], true_items: List[int], k: int) -> float:
    if len(true_items) == 0:
        return 0.0
    predicted_items = predicted_items[:k]
    count_rel_in_top = len(set(true_items) & set(predicted_items))

    return count_rel_in_top / min(k, len(true_items))


def ap_at_k(predicted_items: List[int], true_items: List[int], k: int) -> float:
    if len(true_items) == 0:
        return 0.0

    true_items = set(true_items)
    hits = 0
    sum_precision = 0

    for i in range(len(predicted_items[:k])):
        if predicted_items[i] in true_items:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(k, len(true_items))


def dcg_at_k(relevances: List[int], k: int) -> float:
    if len(relevances) == 0:
        return 0.0

    relevances = np.asarray(relevances)[:k]
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    gains = 2 ** relevances - 1
    dcg = np.sum(gains / discounts)
    return float(dcg)


def ndcg_at_k(relevances: List[int], k: int) -> float:
    if len(relevances) == 0:
        return 0.0

    relevances = np.asarray(relevances)[:k]

    dcg = dcg_at_k(relevances, k)
    ideal = np.sort(relevances)[::-1]
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def coverage(predicted_lists: List[int], total_items: int) -> float:
    if total_items == 0:
        return 0.0

    all_predicted = set()
    for lst in predicted_lists:
        all_predicted.update(lst)

    return len(all_predicted) / total_items


def diversity(predicted_lists: List[List[int]], item_features: dict) -> float:
    """
    Diversity = 1 - среднее intra-list similarity.

    Intra-list similarity = доля пар item'ов с одинаковой фичей (например, категорией).
    Чем меньше одинаковых пар — тем выше Diversity.

    Args:
        predicted_lists: список списков item_id (по одному на запрос)
        item_features: dict {item_id: feature} — например, категория или бренд

    Returns:
        Diversity в [0, 1]. 1 = все item'ы разные, 0 = все одинаковые.
    """
    if not predicted_lists:
        return 0.0

    similarities = []

    for lst in predicted_lists:
        # Пропускаем выдачи, где меньше 2 item'ов — не с чем сравнивать
        if len(lst) < 2:
            continue

        n_pairs = 0
        n_same = 0

        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                n_pairs += 1
                if item_features.get(lst[i]) == item_features.get(lst[j]):
                    n_same += 1

        if n_pairs > 0:
            similarities.append(n_same / n_pairs)

    if not similarities:
        return 0.0

    return 1.0 - float(np.mean(similarities))


def evaluate_ranking(
        predicted_items: Dict[int, List[int]],
        true_items: Dict[int, List[int]],
        relevances: Dict[int, List[int]],
        k: int = 10,
        item_features = None
) -> dict:
    ndcgs, recalls, rrs, hit_rates, aps, precisions = [], [], [], [], [], []
    predicted_lists = []
    for request_id, preds in predicted_items.items():
        if request_id not in true_items:
            continue

        trues = true_items[request_id]
        rels = relevances[request_id]

        if len(trues) == 0:
            continue

        ndcgs.append(ndcg_at_k(rels, k))
        recalls.append(recall_at_k(preds, trues, k))
        rrs.append(rr_at_k(preds, trues, k))
        hit_rates.append(hit_rate_at_k(preds, trues, k))
        aps.append(ap_at_k(preds, trues, k))
        precisions.append(precision_at_k(preds, trues, k))

        predicted_lists.append(preds[:k])

    if not ndcgs:
        return {
            f"ndcg@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            f"map@{k}": 0.0,
            f"precision@{k}": 0.0,
            "n_requests": 0,
        }

    result = {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"recall@{k}": float(np.mean(recalls)),
        f"mrr@{k}": float(np.mean(rrs)),
        f"hit_rate@{k}": float(np.mean(hit_rates)),
        f"map@{k}": float(np.mean(aps)),
        f"precision@{k}": float(np.mean(precisions)),
        "n_requests": len(ndcgs),
    }
    if item_features is not None:
        total_items = len(set(item_features.keys()))
        result[f"coverage@{k}"] = coverage(predicted_lists, total_items)
        result[f"diversity@{k}"] = diversity(predicted_lists, item_features)
    return result


def evaluate_candidates(
        predicted_candidates: Dict[int, List[int]],
        true_items: Dict[int, List[int]],
        k: int = 100,
        total_items=None
) -> dict:
    recalls = []
    hit_rates = []
    rrs = []
    n_candidates_list = []
    all_predicted = set()

    for request_id, candidates in predicted_candidates.items():
        if request_id not in true_items:
            continue

        trues = true_items[request_id]

        if len(trues) == 0:
            continue

        candidates_k = candidates[:k]
        recalls.append(recall_at_k(candidates_k, trues, k))
        hit_rates.append(hit_rate_at_k(candidates_k, trues, k))
        rrs.append(rr_at_k(candidates_k, trues, k))
        n_candidates_list.append(len(candidates_k))

        all_predicted.update(candidates_k)
    if not recalls:
        return {
            f"recall@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            f"mrr@{k}": 0.0,
            "coverage": 0.0,
            "n_requests": 0,
            "avg_candidates": 0.0,
        }

    if total_items is None:
        total_items = len(all_predicted)

    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"hit_rate@{k}": float(np.mean(hit_rates)),
        f"mrr@{k}": float(np.mean(rrs)),
        "coverage": len(all_predicted) / total_items if total_items > 0 else 0.0,
        "n_requests": len(recalls),
        "avg_candidates": float(np.mean(n_candidates_list)),
    }


