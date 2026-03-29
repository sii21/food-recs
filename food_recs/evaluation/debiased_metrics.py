"""Debiased evaluation with stratified sampling by popularity buckets

Addresses the issue where TopPopular wins due to evaluation bias
towards popular items in the test set
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def _compute_popularity_buckets(
    baskets: list[list[int]],
    n_buckets: int = 5,
) -> tuple[dict[int, int], list[tuple[float, float]]]:
    """Assign each item to a popularity bucket based on frequency

    Args:
        baskets: Training baskets used to compute item popularity
        n_buckets: Number of popularity buckets (quintiles by default)

    Returns:
        Tuple of (item_to_bucket, bucket_ranges)
        item_to_bucket maps item_id -> bucket_index (0 = least popular)
        bucket_ranges is list of (min_freq, max_freq) per bucket
    """
    item_counts: dict[int, int] = defaultdict(int)
    for basket in baskets:
        for item in basket:
            item_counts[item] += 1

    if not item_counts:
        return {}, []

    counts = np.array(list(item_counts.values()), dtype=float)
    percentiles = np.linspace(0, 100, n_buckets + 1)
    thresholds = np.percentile(counts, percentiles)

    item_to_bucket: dict[int, int] = {}
    for item_id, count in item_counts.items():
        for b in range(n_buckets):
            if count <= thresholds[b + 1] or b == n_buckets - 1:
                item_to_bucket[item_id] = b
                break

    bucket_ranges = [
        (float(thresholds[i]), float(thresholds[i + 1]))
        for i in range(n_buckets)
    ]
    return item_to_bucket, bucket_ranges


def stratified_leave_one_out(
    test_data: list[tuple[list[int], int, int | None]],
    train_baskets: list[list[int]],
    n_buckets: int = 5,
    samples_per_bucket: int | None = None,
    seed: int = 42,
) -> tuple[list[tuple[list[int], int, int | None]], dict[int, int]]:
    """Create stratified leave-one-out test data balanced across popularity buckets

    Args:
        test_data: Original L1O data (input_basket, held_out_item, profile_id)
        train_baskets: Training baskets for popularity computation
        n_buckets: Number of popularity buckets
        samples_per_bucket: Max samples per bucket (None = use min bucket size)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (stratified_test_data, item_to_bucket)
    """
    item_to_bucket, bucket_ranges = _compute_popularity_buckets(train_baskets, n_buckets)

    if not item_to_bucket:
        return test_data, {}

    # Group test examples by held-out item's popularity bucket
    bucket_examples: dict[int, list[tuple[list[int], int, int | None]]] = defaultdict(list)
    unknown_bucket = []

    for entry in test_data:
        held_out_item = entry[1]
        if held_out_item in item_to_bucket:
            bucket = item_to_bucket[held_out_item]
            bucket_examples[bucket].append(entry)
        else:
            unknown_bucket.append(entry)

    # Determine samples per bucket
    bucket_sizes = {b: len(examples) for b, examples in bucket_examples.items()}
    if samples_per_bucket is None:
        samples_per_bucket = min(bucket_sizes.values()) if bucket_sizes else 0

    print(f"Stratified L1O: {n_buckets} popularity buckets")
    for b in range(n_buckets):
        lo, hi = bucket_ranges[b] if b < len(bucket_ranges) else (0, 0)
        n = bucket_sizes.get(b, 0)
        sampled = min(n, samples_per_bucket)
        print(f"  Bucket {b} (freq {lo:.0f}-{hi:.0f}): {n:,} examples -> {sampled:,} sampled")
    if unknown_bucket:
        print(f"  Unknown (new items): {len(unknown_bucket):,} examples (included as-is)")

    # Stratified sampling
    rng = np.random.default_rng(seed)
    stratified = []
    for b in range(n_buckets):
        examples = bucket_examples.get(b, [])
        if not examples:
            continue
        n_sample = min(len(examples), samples_per_bucket)
        indices = rng.choice(len(examples), size=n_sample, replace=False)
        stratified.extend(examples[i] for i in indices)

    # Include unknown-bucket items (new items not in train)
    stratified.extend(unknown_bucket)

    rng.shuffle(stratified)
    print(f"  Total stratified samples: {len(stratified):,}")
    return stratified, item_to_bucket


class DebiasedEvaluator:
    """Evaluator that reports metrics per popularity bucket and overall debiased metrics

    Computes standard Hit@K and MRR but also breaks down results by item
    popularity to detect bias towards popular items
    """

    def __init__(
        self,
        train_baskets: list[list[int]],
        n_buckets: int = 5,
    ):
        self.n_buckets = n_buckets
        self.item_to_bucket, self.bucket_ranges = _compute_popularity_buckets(
            train_baskets, n_buckets
        )

    def evaluate(
        self,
        model,
        test_data: list[tuple],
        k_values: list[int],
        split_name: str = "test",
        n_items_total: int | None = None,
    ) -> dict[str, float]:
        """Evaluate model with per-bucket and debiased metrics

        Args:
            model: Trained recommender model with .recommend(basket, k) method
            test_data: List of (input_basket, held_out_item, ...) tuples
            k_values: List of K values for Hit@K
            split_name: Prefix for metric names
            n_items_total: Total number of unique items (for coverage)

        Returns:
            Dict with standard metrics + per-bucket metrics + debiased metrics
        """
        if not test_data:
            return {f"{split_name}_hit@{k}": 0.0 for k in k_values} | {
                f"{split_name}_mrr": 0.0
            }

        # Per-bucket accumulators
        bucket_hits: dict[int, dict[int, int]] = {
            b: {k: 0 for k in k_values} for b in range(self.n_buckets)
        }
        bucket_mrr: dict[int, float] = defaultdict(float)
        bucket_counts: dict[int, int] = defaultdict(int)

        # Overall accumulators
        results = {f"{split_name}_hit@{k}": 0.0 for k in k_values}
        mrr_sum = 0.0
        unique_recommended: set[int] = set()
        total_reco_time = 0.0

        for entry in tqdm(test_data, desc=f"Debiased eval ({split_name})"):
            input_basket, held_out_item = entry[0], entry[1]

            t0 = time.perf_counter()
            recs = model.recommend(input_basket, k=max(k_values))
            total_reco_time += time.perf_counter() - t0
            unique_recommended.update(recs[:max(k_values)])

            bucket = self.item_to_bucket.get(held_out_item, -1)

            if held_out_item in recs:
                rank = recs.index(held_out_item) + 1
                mrr_sum += 1 / rank
                if bucket >= 0:
                    bucket_mrr[bucket] += 1 / rank

            for k in k_values:
                if held_out_item in recs[:k]:
                    results[f"{split_name}_hit@{k}"] += 1
                    if bucket >= 0:
                        bucket_hits[bucket][k] += 1

            if bucket >= 0:
                bucket_counts[bucket] += 1

        n_test = len(test_data)
        for key in results:
            results[key] /= n_test
        results[f"{split_name}_mrr"] = mrr_sum / n_test
        results[f"{split_name}_avg_latency_ms"] = (total_reco_time / n_test) * 1000.0
        if n_items_total and n_items_total > 0:
            results[f"{split_name}_coverage"] = len(unique_recommended) / n_items_total
        else:
            results[f"{split_name}_coverage"] = 0.0

        # Per-bucket metrics
        for b in range(self.n_buckets):
            n_b = bucket_counts.get(b, 0)
            if n_b == 0:
                continue
            for k in k_values:
                results[f"{split_name}_bucket{b}_hit@{k}"] = bucket_hits[b][k] / n_b
            results[f"{split_name}_bucket{b}_mrr"] = bucket_mrr[b] / n_b

        # Debiased metrics: macro-average across buckets (equal weight per bucket)
        active_buckets = [b for b in range(self.n_buckets) if bucket_counts.get(b, 0) > 0]
        if active_buckets:
            for k in k_values:
                bucket_hit_rates = [
                    bucket_hits[b][k] / bucket_counts[b]
                    for b in active_buckets
                ]
                results[f"{split_name}_debiased_hit@{k}"] = float(np.mean(bucket_hit_rates))

            bucket_mrr_rates = [
                bucket_mrr[b] / bucket_counts[b]
                for b in active_buckets
            ]
            results[f"{split_name}_debiased_mrr"] = float(np.mean(bucket_mrr_rates))

        return results

    def print_bucket_summary(self, results: dict[str, float], split_name: str = "test") -> None:
        """Print per-bucket metrics summary"""
        print(f"\n{'=' * 60}")
        print(f"Debiased Evaluation Summary ({split_name})")
        print(f"{'=' * 60}")

        print(f"\n{'Bucket':<12} {'Freq Range':<16} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}")
        print("-" * 60)

        for b in range(self.n_buckets):
            if f"{split_name}_bucket{b}_hit@5" not in results:
                continue
            lo, hi = self.bucket_ranges[b] if b < len(self.bucket_ranges) else (0, 0)
            h5 = results.get(f"{split_name}_bucket{b}_hit@5", 0)
            h10 = results.get(f"{split_name}_bucket{b}_hit@10", 0)
            mrr = results.get(f"{split_name}_bucket{b}_mrr", 0)
            print(f"  {b:<10} {lo:>6.0f}-{hi:<6.0f}  {h5:>8.4f} {h10:>8.4f} {mrr:>8.4f}")

        # Debiased totals
        dh5 = results.get(f"{split_name}_debiased_hit@5", 0)
        dh10 = results.get(f"{split_name}_debiased_hit@10", 0)
        dmrr = results.get(f"{split_name}_debiased_mrr", 0)
        print("-" * 60)
        print(f"  {'Debiased':<10} {'(macro avg)':<16} {dh5:>8.4f} {dh10:>8.4f} {dmrr:>8.4f}")

        # Standard metrics for comparison
        h5 = results.get(f"{split_name}_hit@5", 0)
        h10 = results.get(f"{split_name}_hit@10", 0)
        mrr = results.get(f"{split_name}_mrr", 0)
        print(f"  {'Standard':<10} {'(micro avg)':<16} {h5:>8.4f} {h10:>8.4f} {mrr:>8.4f}")
