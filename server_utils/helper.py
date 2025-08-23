from typing import List, Dict, Tuple


def unlearning_weighted_loss_avg(
    results: List[Tuple[int, float, int, float]]
) -> Dict[str, float]:
    """
    Aggregate unlearning losses (impair and repair losses) computed from multiple clients,
    each with potentially different numbers of examples.
    """
    total_impair_examples = sum(impair_examples for impair_examples, _, _, _ in results)
    total_repair_examples = sum(repair_examples for _, _, repair_examples, _ in results)

    weighted_impair_loss = (
        sum(
            impair_examples * impair_loss
            for impair_examples, impair_loss, _, _ in results
        )
        / total_impair_examples
        if total_impair_examples > 0
        else 0.0
    )
    weighted_repair_loss = (
        sum(
            repair_examples * repair_loss
            for _, _, repair_examples, repair_loss in results
        )
        / total_repair_examples
        if total_repair_examples > 0
        else 0.0
    )

    return (
        weighted_impair_loss,
        weighted_repair_loss,
    )


def multi_weighted_avg(
    results: List[Dict[str, float]], metric_pairs: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Compute weighted averages for multiple metrics given a list of dictionaries.
    """
    aggregated = {}
    for count_key, metric_key in metric_pairs:
        # Filter results that have both keys
        valid_results = [
            result for result in results 
            if count_key in result and metric_key in result
        ]
        
        if not valid_results:
            aggregated[metric_key] = 0.0
            continue
            
        total_count = sum(result[count_key] for result in valid_results)
        weighted_sum = sum(
            result[count_key] * result[metric_key] for result in valid_results
        )
        aggregated[metric_key] = weighted_sum / total_count if total_count > 0 else 0.0
    return aggregated